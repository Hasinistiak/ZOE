from __future__ import annotations

import asyncio
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from typing import Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from backend.zoe import (
    extract_response_text,
    get_zoe_snapshot,
    interrupt_zoe,
    mute_zoe,
    run_zoe,
    subscribe_runtime_events,
    unsubscribe_runtime_events,
    unmute_zoe,
)

# IMPORTANT:
# Change this import if your PocketTTS module has a different filename.
from backend.speech.tts import generate_voice_file


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

TELEGRAM_ENABLED = (
    os.getenv(
        "TELEGRAM_ENABLED",
        "false",
    )
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "",
).strip()

_ALLOWED_USER_RAW = os.getenv(
    "TELEGRAM_ALLOWED_USER_ID",
    "",
).strip()

try:
    TELEGRAM_ALLOWED_USER_ID: Optional[int] = (
        int(_ALLOWED_USER_RAW)
        if _ALLOWED_USER_RAW
        else None
    )
except ValueError:
    TELEGRAM_ALLOWED_USER_ID = None


# ============================================================
# TELEGRAM VOICE
# ============================================================

TELEGRAM_VOICE_ENABLED = (
    os.getenv(
        "TELEGRAM_VOICE_ENABLED",
        "true",
    )
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

TELEGRAM_VOICE_PROACTIVE = (
    os.getenv(
        "TELEGRAM_VOICE_PROACTIVE",
        "true",
    )
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

TELEGRAM_VOICE_FFMPEG = (
    os.getenv(
        "TELEGRAM_VOICE_FFMPEG",
        "ffmpeg",
    )
    .strip()
    or "ffmpeg"
)

TELEGRAM_VOICE_TIMEOUT = max(
    10.0,
    float(
        os.getenv(
            "TELEGRAM_VOICE_TIMEOUT",
            "120",
        )
    ),
)


# ============================================================
# MESSAGE LIMITS
# ============================================================

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_MAX_INPUT_LENGTH = 12000


# ============================================================
# RESPONSE TIMEOUT
# ============================================================

try:
    TELEGRAM_RESPONSE_TIMEOUT = max(
        5.0,
        float(
            os.getenv(
                "ZOE_TELEGRAM_RESPONSE_TIMEOUT",
                "300",
            )
        ),
    )
except (TypeError, ValueError):
    TELEGRAM_RESPONSE_TIMEOUT = 300.0


# ============================================================
# PROACTIVE EVENTS
# ============================================================

PROACTIVE_OUTPUT_TYPES = {
    "notification",
    "reminder",
}


LOGGER = logging.getLogger(
    "zoe.telegram"
)


# ============================================================
# GLOBAL BOT STATE
# ============================================================

_bot_thread: Optional[
    threading.Thread
] = None

_bot_loop: Optional[
    asyncio.AbstractEventLoop
] = None

_bot_application: Optional[
    Application
] = None

_bot_started = threading.Event()
_bot_stop_requested = threading.Event()
_bot_state_lock = threading.RLock()


# ============================================================
# PROACTIVE EVENT STATE
# ============================================================

_proactive_thread: Optional[
    threading.Thread
] = None

_proactive_stop_requested = threading.Event()

_proactive_state_lock = threading.RLock()

_proactive_subscriber_id: Optional[str] = None

_proactive_queue: Optional[
    queue.Queue
] = None


# ============================================================
# LOGGING
# ============================================================

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


# ============================================================
# SECURITY
# ============================================================

def _is_authorized(
    update: Update,
) -> bool:
    if TELEGRAM_ALLOWED_USER_ID is None:
        return False

    user = update.effective_user

    if user is None:
        return False

    return (
        user.id
        == TELEGRAM_ALLOWED_USER_ID
    )


def _log_unauthorized(
    update: Update,
) -> None:
    user = update.effective_user

    if user is None:
        LOGGER.warning(
            "Telegram unauthorized request "
            "from unknown user."
        )
        return

    LOGGER.warning(
        "Telegram unauthorized user: "
        "id=%s username=%s name=%s",
        user.id,
        user.username,
        user.full_name,
    )


# ============================================================
# MESSAGE HELPERS
# ============================================================

def _split_message(
    text: str,
    max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH,
) -> list[str]:

    text = (
        text
        or ""
    ).strip()

    if not text:
        return []

    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []

    while len(text) > max_length:

        split_at = text.rfind(
            "\n",
            0,
            max_length,
        )

        if split_at < 1000:
            split_at = text.rfind(
                " ",
                0,
                max_length,
            )

        if split_at < 1000:
            split_at = max_length

        chunks.append(
            text[:split_at].rstrip()
        )

        text = (
            text[split_at:]
            .lstrip()
        )

    if text:
        chunks.append(text)

    return chunks


async def _reply(
    update: Update,
    text: str,
) -> None:

    message = (
        update.effective_message
    )

    if message is None:
        return

    chunks = _split_message(
        text
    )

    for chunk in chunks:
        await message.reply_text(
            chunk
        )


# ============================================================
# VOICE FILE CONVERSION
# ============================================================

def _wav_to_ogg(
    wav_path: str,
) -> str | None:
    """
    Convert WAV -> OGG/Opus for Telegram Voice Messages.
    """

    if not os.path.isfile(
        wav_path
    ):
        return None

    ffmpeg = shutil.which(
        TELEGRAM_VOICE_FFMPEG
    )

    if ffmpeg is None:
        LOGGER.error(
            "FFmpeg was not found. "
            "Telegram voice messages require FFmpeg."
        )
        return None

    fd, ogg_path = tempfile.mkstemp(
        prefix="zoe_voice_",
        suffix=".ogg",
    )

    os.close(fd)

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        wav_path,
        "-c:a",
        "libopus",
        "-b:a",
        "48k",
        "-vbr",
        "on",
        "-application",
        "voip",
        ogg_path,
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=TELEGRAM_VOICE_TIMEOUT,
            check=False,
        )

        if result.returncode != 0:

            error = (
                result.stderr
                .decode(
                    "utf-8",
                    errors="replace",
                )
                .strip()
            )

            LOGGER.error(
                "FFmpeg voice conversion failed: %s",
                error or "unknown error",
            )

            try:
                os.remove(
                    ogg_path
                )
            except OSError:
                pass

            return None

        return ogg_path

    except subprocess.TimeoutExpired:

        LOGGER.error(
            "FFmpeg voice conversion timed out."
        )

        try:
            os.remove(
                ogg_path
            )
        except OSError:
            pass

        return None

    except Exception:

        LOGGER.exception(
            "Unexpected FFmpeg error."
        )

        try:
            os.remove(
                ogg_path
            )
        except OSError:
            pass

        return None


def _generate_telegram_voice(
    text: str,
) -> str | None:
    """
    Generate ZOE's PocketTTS voice and convert
    it into Telegram-compatible OGG/Opus.
    """

    text = (
        text
        or ""
    ).strip()

    if not text:
        return None

    wav_path = None
    ogg_path = None

    try:

        wav_path = generate_voice_file(
            text
        )

        if not wav_path:
            LOGGER.warning(
                "PocketTTS produced no voice file."
            )
            return None

        ogg_path = _wav_to_ogg(
            wav_path
        )

        if not ogg_path:
            return None

        return ogg_path

    except Exception:

        LOGGER.exception(
            "Failed to generate Telegram voice."
        )

        if ogg_path:
            try:
                os.remove(
                    ogg_path
                )
            except OSError:
                pass

        return None

    finally:

        if wav_path:
            try:
                os.remove(
                    wav_path
                )
            except OSError:
                pass


# ============================================================
# /START
# ============================================================

async def _start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not _is_authorized(update):

        _log_unauthorized(
            update
        )

        message = (
            update.effective_message
        )

        if message is not None:
            await message.reply_text(
                "Access denied."
            )

        return

    await _reply(
        update,
        (
            "Online, Sir.\n\n"
            "I'm connected to your ZOE runtime.\n\n"
            "Send me a normal message and "
            "I'll process it through ZOE.\n\n"
            "You can also send me a voice message. "
            "I'll transcribe it and reply with a "
            "Telegram voice message.\n\n"
            "For longer agent tasks, I'll wait "
            "for the actual final result before "
            "replying.\n\n"
            "Proactive notifications are also "
            "enabled.\n\n"
            "Commands:\n"
            "/status — ZOE status\n"
            "/mute — mute ZOE voice\n"
            "/unmute — unmute ZOE voice\n"
            "/stop — stop ZOE speech\n"
            "/help — show commands"
        ),
    )


# ============================================================
# /HELP
# ============================================================

async def _help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not _is_authorized(update):

        _log_unauthorized(
            update
        )

        return

    await _reply(
        update,
        (
            "ZOE Telegram Interface\n\n"
            "Text:\n"
            "Send any message to ZOE.\n\n"
            "Voice:\n"
            "Send a Telegram voice message. "
            "ZOE will transcribe it and reply "
            "with a voice message.\n\n"
            "Agent requests:\n"
            "ZOE waits for the final synthesized "
            "answer before replying.\n\n"
            "Proactive:\n"
            "ZOE can send reminders and status "
            "notifications automatically.\n\n"
            "Commands:\n"
            "/start — Start the interface\n"
            "/help — Show this help\n"
            "/status — Show ZOE status\n"
            "/mute — Mute ZOE voice output\n"
            "/unmute — Enable ZOE voice output\n"
            "/stop — Stop current ZOE speech"
        ),
    )


# ============================================================
# /STATUS
# ============================================================

async def _status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not _is_authorized(update):

        _log_unauthorized(
            update
        )

        return

    try:

        snapshot = get_zoe_snapshot()

        online = snapshot.get(
            "online",
            False,
        )

        state = snapshot.get(
            "state",
            "unknown",
        )

        listening = snapshot.get(
            "listening",
            False,
        )

        thinking = snapshot.get(
            "thinking",
            False,
        )

        speaking = snapshot.get(
            "speaking",
            False,
        )

        processing = snapshot.get(
            "processing",
            False,
        )

        muted = snapshot.get(
            "muted",
            False,
        )

        queued = snapshot.get(
            "queued_requests",
            0,
        )

        proactive_running = (
            is_proactive_notifications_running()
        )

        text = (
            "ZOE STATUS\n\n"
            f"Online: "
            f"{'YES' if online else 'NO'}\n"
            f"State: {state}\n"
            f"Listening: "
            f"{'YES' if listening else 'NO'}\n"
            f"Thinking: "
            f"{'YES' if thinking else 'NO'}\n"
            f"Speaking: "
            f"{'YES' if speaking else 'NO'}\n"
            f"Processing: "
            f"{'YES' if processing else 'NO'}\n"
            f"Muted: "
            f"{'YES' if muted else 'NO'}\n"
            f"Queued requests: {queued}\n"
            f"Telegram proactive: "
            f"{'YES' if proactive_running else 'NO'}\n"
            f"Telegram voice: "
            f"{'YES' if TELEGRAM_VOICE_ENABLED else 'NO'}"
        )

        await _reply(
            update,
            text,
        )

    except Exception:

        LOGGER.exception(
            "Failed to get ZOE status."
        )

        await _reply(
            update,
            "I couldn't retrieve the current ZOE status.",
        )


# ============================================================
# /MUTE
# ============================================================

async def _mute_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not _is_authorized(update):

        _log_unauthorized(
            update
        )

        return

    try:

        mute_zoe()

        await _reply(
            update,
            "Voice output muted.",
        )

    except Exception:

        LOGGER.exception(
            "Failed to mute ZOE."
        )

        await _reply(
            update,
            "I couldn't mute ZOE.",
        )


# ============================================================
# /UNMUTE
# ============================================================

async def _unmute_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not _is_authorized(update):

        _log_unauthorized(
            update
        )

        return

    try:

        unmute_zoe()

        await _reply(
            update,
            "Voice output restored.",
        )

    except Exception:

        LOGGER.exception(
            "Failed to unmute ZOE."
        )

        await _reply(
            update,
            "I couldn't unmute ZOE.",
        )


# ============================================================
# /STOP
# ============================================================

async def _stop_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not _is_authorized(update):

        _log_unauthorized(
            update
        )

        return

    try:

        interrupt_zoe()

        await _reply(
            update,
            "Stopped.",
        )

    except Exception:

        LOGGER.exception(
            "Failed to stop ZOE speech."
        )

        await _reply(
            update,
            "I couldn't stop the current speech.",
        )


# ============================================================
# SEND TELEGRAM TEXT
# ============================================================

async def _send_message_async(
    text: str,
) -> bool:

    with _bot_state_lock:
        application = (
            _bot_application
        )

    if application is None:

        LOGGER.warning(
            "Cannot send Telegram message: "
            "bot is not running."
        )

        return False

    if TELEGRAM_ALLOWED_USER_ID is None:
        return False

    chunks = _split_message(
        text
    )

    if not chunks:
        return False

    try:

        for chunk in chunks:

            await application.bot.send_message(
                chat_id=TELEGRAM_ALLOWED_USER_ID,
                text=chunk,
            )

        return True

    except Exception:

        LOGGER.exception(
            "Failed to send Telegram message."
        )

        return False


def send_telegram_message(
    text: str,
    timeout: float = 15.0,
) -> bool:

    if not TELEGRAM_ENABLED:
        return False

    text = (
        text
        or ""
    ).strip()

    if not text:
        return False

    with _bot_state_lock:
        loop = _bot_loop

    if (
        loop is None
        or not loop.is_running()
    ):

        LOGGER.warning(
            "Cannot send Telegram message: "
            "event loop is not running."
        )

        return False

    try:

        future = (
            asyncio.run_coroutine_threadsafe(
                _send_message_async(text),
                loop,
            )
        )

        return bool(
            future.result(
                timeout=timeout
            )
        )

    except Exception:

        LOGGER.exception(
            "Telegram outbound message failed."
        )

        return False


# ============================================================
# SEND TELEGRAM VOICE
# ============================================================

async def _send_voice_async(
    text: str,
) -> bool:

    if not TELEGRAM_VOICE_ENABLED:
        return False

    with _bot_state_lock:
        application = (
            _bot_application
        )

    if application is None:

        LOGGER.warning(
            "Cannot send Telegram voice: "
            "bot is not running."
        )

        return False

    if TELEGRAM_ALLOWED_USER_ID is None:
        return False

    voice_path = None

    try:

        voice_path = await asyncio.to_thread(
            _generate_telegram_voice,
            text,
        )

        if not voice_path:
            return False

        with open(
            voice_path,
            "rb",
        ) as voice_file:

            await application.bot.send_voice(
                chat_id=TELEGRAM_ALLOWED_USER_ID,
                voice=voice_file,
            )

        LOGGER.info(
            "ZOE voice -> Telegram delivered."
        )

        return True

    except Exception:

        LOGGER.exception(
            "Failed to send Telegram voice."
        )

        return False

    finally:

        if voice_path:

            try:
                os.remove(
                    voice_path
                )
            except OSError:
                pass


def send_telegram_voice(
    text: str,
    timeout: float = TELEGRAM_VOICE_TIMEOUT,
) -> bool:
    """
    Thread-safe synchronous Telegram voice API.

    Generates PocketTTS audio, converts it to
    OGG/Opus and sends it as a native Telegram VM.
    """

    if not TELEGRAM_ENABLED:
        return False

    if not TELEGRAM_VOICE_ENABLED:
        return False

    text = (
        text
        or ""
    ).strip()

    if not text:
        return False

    with _bot_state_lock:
        loop = _bot_loop

    if (
        loop is None
        or not loop.is_running()
    ):

        LOGGER.warning(
            "Cannot send Telegram voice: "
            "event loop is not running."
        )

        return False

    try:

        future = (
            asyncio.run_coroutine_threadsafe(
                _send_voice_async(text),
                loop,
            )
        )

        return bool(
            future.result(
                timeout=timeout
            )
        )

    except Exception:

        LOGGER.exception(
            "Telegram outbound voice failed."
        )

        return False


# ============================================================
# NORMAL TEXT MESSAGE
# ============================================================

async def _message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not _is_authorized(update):

        _log_unauthorized(
            update
        )

        return

    message = (
        update.effective_message
    )

    if message is None:
        return

    text = (
        message.text
        or ""
    ).strip()

    if not text:
        return

    if len(text) > TELEGRAM_MAX_INPUT_LENGTH:

        await _reply(
            update,
            (
                "That message is too long. "
                "Please split it into smaller messages."
            ),
        )

        return

    LOGGER.info(
        "Telegram -> ZOE: %s",
        text[:200],
    )

    try:

        response = await asyncio.to_thread(
            run_zoe,
            text,
            False,
            True,
            TELEGRAM_RESPONSE_TIMEOUT,
        )

        response_text = extract_response_text(
            response
        )

        if response_text:

            LOGGER.info(
                "ZOE -> Telegram: %s",
                response_text[:200],
            )

            # Always send text.
            await _reply(
                update,
                response_text,
            )

            # Then send native VM.
            if TELEGRAM_VOICE_ENABLED:

                voice_sent = await asyncio.to_thread(
                    send_telegram_voice,
                    response_text,
                )

                if not voice_sent:

                    LOGGER.warning(
                        "Telegram voice reply failed; "
                        "text reply was already delivered."
                    )

            return

        LOGGER.warning(
            "ZOE returned no usable response "
            "for Telegram request."
        )

        await _reply(
            update,
            (
                "I'm still working on that. "
                "I'll send the result when it's ready."
            ),
        )

    except Exception as exc:

        LOGGER.exception(
            "Telegram -> ZOE request failed."
        )

        await _reply(
            update,
            (
                "I couldn't complete that request.\n\n"
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# TELEGRAM VOICE MESSAGE INPUT
# ============================================================

async def _voice_message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not _is_authorized(update):

        _log_unauthorized(
            update
        )

        return

    message = (
        update.effective_message
    )

    if message is None:
        return

    voice = message.voice

    if voice is None:
        return

    LOGGER.info(
        "Telegram voice -> ZOE | duration=%ss",
        voice.duration,
    )

    temp_dir = tempfile.mkdtemp(
        prefix="zoe_telegram_voice_"
    )

    input_path = os.path.join(
        temp_dir,
        "input.ogg",
    )

    try:

        # ====================================================
        # DOWNLOAD VOICE MESSAGE
        # ====================================================

        telegram_file = await context.bot.get_file(
            voice.file_id
        )

        await telegram_file.download_to_drive(
            input_path
        )

        # ====================================================
        # STT
        # ====================================================

        from backend.speech.stt import (
            transcribe_audio_file,
        )

        transcript = await asyncio.to_thread(
            transcribe_audio_file,
            input_path,
        )

        transcript = (
            transcript
            or ""
        ).strip()

        if not transcript:

            await _reply(
                update,
                "I couldn't understand that voice message.",
            )

            return

        # ====================================================
        # IMPORTANT:
        # DO NOT SEND THE TRANSCRIPT BACK TO TELEGRAM.
        #
        # The transcript is internal and goes directly
        # into ZOE.
        # ====================================================

        LOGGER.info(
            "Telegram VM transcribed successfully."
        )

        # ====================================================
        # ZOE
        # ====================================================

        response = await asyncio.to_thread(
            run_zoe,
            transcript,
            False,
            True,
            TELEGRAM_RESPONSE_TIMEOUT,
        )

        response_text = extract_response_text(
            response
        )

        if not response_text:

            await _reply(
                update,
                (
                    "I'm still working on that. "
                    "I'll send the result when it's ready."
                ),
            )

            return

        # ====================================================
        # TEXT RESPONSE
        # ====================================================

        await _reply(
            update,
            response_text,
        )

        # ====================================================
        # VOICE RESPONSE
        # ====================================================

        if TELEGRAM_VOICE_ENABLED:

            voice_sent = await asyncio.to_thread(
                send_telegram_voice,
                response_text,
            )

            if not voice_sent:

                LOGGER.warning(
                    "Telegram VM voice response failed."
                )

    except Exception as exc:

        LOGGER.exception(
            "Telegram voice processing failed."
        )

        await _reply(
            update,
            (
                "I couldn't process that voice message.\n\n"
                f"{type(exc).__name__}: {exc}"
            ),
        )

    finally:

        try:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

        except Exception:

            pass


# ============================================================
# ERROR HANDLER
# ============================================================

async def _error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    LOGGER.error(
        "Telegram error: %s",
        context.error,
        exc_info=context.error,
    )


# ============================================================
# APPLICATION
# ============================================================

def _build_application() -> Application:

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured."
        )

    if TELEGRAM_ALLOWED_USER_ID is None:

        raise RuntimeError(
            "TELEGRAM_ALLOWED_USER_ID is not "
            "configured or is invalid."
        )

    application = (
        Application.builder()
        .token(
            TELEGRAM_BOT_TOKEN
        )
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            _start_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            _help_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "status",
            _status_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "mute",
            _mute_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "unmute",
            _unmute_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "stop",
            _stop_command,
        )
    )

    # Voice messages MUST be registered separately.
    application.add_handler(
        MessageHandler(
            filters.VOICE,
            _voice_message_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            _message_handler,
        )
    )

    application.add_error_handler(
        _error_handler
    )

    return application


# ============================================================
# TELEGRAM ASYNC MAIN
# ============================================================

async def _telegram_main() -> None:

    global _bot_application

    application = (
        _build_application()
    )

    with _bot_state_lock:

        _bot_application = (
            application
        )

    try:

        await application.initialize()

        await application.start()

        if application.updater is None:

            raise RuntimeError(
                "Telegram Application "
                "updater is unavailable."
            )

        await application.updater.start_polling(
            drop_pending_updates=True,
        )

        start_proactive_notifications()

        _bot_started.set()

        LOGGER.info(
            "Telegram bot polling started."
        )

        LOGGER.info(
            "Telegram proactive notification "
            "bridge started."
        )

        LOGGER.info(
            "Telegram voice messages: %s",
            (
                "enabled"
                if TELEGRAM_VOICE_ENABLED
                else "disabled"
            ),
        )

        while not _bot_stop_requested.is_set():

            await asyncio.sleep(
                0.5
            )

    finally:

        LOGGER.info(
            "Stopping Telegram bot..."
        )

        stop_proactive_notifications(
            timeout=5.0
        )

        try:

            if (
                application.updater
                is not None
            ):

                await application.updater.stop()

        except Exception:

            LOGGER.exception(
                "Failed to stop Telegram updater."
            )

        try:

            await application.stop()

        except Exception:

            LOGGER.exception(
                "Failed to stop Telegram application."
            )

        try:

            await application.shutdown()

        except Exception:

            LOGGER.exception(
                "Failed to shutdown Telegram application."
            )

        with _bot_state_lock:

            _bot_application = None

        LOGGER.info(
            "Telegram bot stopped."
        )


# ============================================================
# BOT THREAD
# ============================================================

def _telegram_thread_main() -> None:

    global _bot_loop

    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(
        loop
    )

    with _bot_state_lock:

        _bot_loop = loop

    try:

        loop.run_until_complete(
            _telegram_main()
        )

    except Exception:

        LOGGER.exception(
            "Telegram bot crashed."
        )

    finally:

        with _bot_state_lock:

            _bot_loop = None

        try:

            loop.close()

        except Exception:

            LOGGER.exception(
                "Failed to close Telegram "
                "event loop."
            )

        _bot_started.clear()


# ============================================================
# START
# ============================================================

def start_telegram_bot() -> bool:

    global _bot_thread

    if not TELEGRAM_ENABLED:

        LOGGER.info(
            "Telegram integration disabled."
        )

        return False

    if not TELEGRAM_BOT_TOKEN:

        LOGGER.error(
            "Telegram enabled but "
            "TELEGRAM_BOT_TOKEN is missing."
        )

        return False

    if TELEGRAM_ALLOWED_USER_ID is None:

        LOGGER.error(
            "Telegram enabled but "
            "TELEGRAM_ALLOWED_USER_ID is "
            "missing or invalid."
        )

        return False

    if (
        TELEGRAM_VOICE_ENABLED
        and shutil.which(
            TELEGRAM_VOICE_FFMPEG
        ) is None
    ):

        LOGGER.error(
            "Telegram voice is enabled but "
            "FFmpeg was not found: %s",
            TELEGRAM_VOICE_FFMPEG,
        )

        return False

    with _bot_state_lock:

        if (
            _bot_thread is not None
            and _bot_thread.is_alive()
        ):

            LOGGER.info(
                "Telegram bot already running."
            )

            return True

        _bot_stop_requested.clear()
        _bot_started.clear()

        _bot_thread = threading.Thread(
            target=_telegram_thread_main,
            name="ZOE-Telegram",
            daemon=True,
        )

        _bot_thread.start()

    started = _bot_started.wait(
        timeout=15.0
    )

    if not started:

        LOGGER.warning(
            "Telegram bot thread started but "
            "polling has not confirmed startup yet."
        )

    return True


# ============================================================
# STOP
# ============================================================

def stop_telegram_bot(
    timeout: float = 10.0,
) -> None:

    global _bot_thread

    with _bot_state_lock:

        thread = _bot_thread

        if thread is None:
            return

        if not thread.is_alive():

            _bot_thread = None
            return

        _bot_stop_requested.set()

        loop = _bot_loop

    if loop is not None:

        try:

            loop.call_soon_threadsafe(
                lambda: None
            )

        except Exception:

            LOGGER.exception(
                "Failed to wake Telegram "
                "event loop."
            )

    thread.join(
        timeout=timeout
    )

    if thread.is_alive():

        LOGGER.warning(
            "Telegram bot thread did not stop "
            "within %.1fs.",
            timeout,
        )

    with _bot_state_lock:

        _bot_thread = None


# ============================================================
# STATUS
# ============================================================

def is_telegram_running() -> bool:

    with _bot_state_lock:

        return bool(
            _bot_thread
            and _bot_thread.is_alive()
            and _bot_started.is_set()
        )


# ============================================================
# PROACTIVE TELEGRAM
# ============================================================

def _extract_proactive_text(
    event: object,
) -> Optional[str]:

    if not isinstance(
        event,
        dict,
    ):
        return None

    if event.get(
        "type"
    ) != "output":

        return None

    data = event.get(
        "data",
        {},
    )

    if not isinstance(
        data,
        dict,
    ):
        return None

    output_type = str(
        data.get(
            "type",
            "",
        )
    ).strip().lower()

    if output_type not in PROACTIVE_OUTPUT_TYPES:
        return None

    text = str(
        data.get(
            "text",
            "",
        )
        or ""
    ).strip()

    if not text:
        return None

    return text


def _proactive_event_worker() -> None:

    LOGGER.info(
        "ZOE -> Telegram proactive "
        "event worker started."
    )

    while not _proactive_stop_requested.is_set():

        with _proactive_state_lock:

            event_queue = (
                _proactive_queue
            )

        if event_queue is None:
            break

        try:

            event = event_queue.get(
                timeout=0.5
            )

        except queue.Empty:

            continue

        try:

            text = (
                _extract_proactive_text(
                    event
                )
            )

            if not text:
                continue

            event_type = (
                event.get(
                    "data",
                    {},
                ).get(
                    "type",
                    "unknown",
                )
                if isinstance(
                    event,
                    dict,
                )
                else "unknown"
            )

            LOGGER.info(
                "ZOE proactive -> Telegram "
                "| type=%s | text=%s",
                event_type,
                text[:200],
            )

            # Text notification.
            text_success = send_telegram_message(
                text
            )

            if text_success:

                LOGGER.info(
                    "ZOE proactive text "
                    "delivered to Telegram."
                )

            # Voice notification.
            if (
                text_success
                and TELEGRAM_VOICE_ENABLED
                and TELEGRAM_VOICE_PROACTIVE
            ):

                voice_success = (
                    send_telegram_voice(
                        text
                    )
                )

                if voice_success:

                    LOGGER.info(
                        "ZOE proactive voice "
                        "delivered to Telegram."
                    )

                else:

                    LOGGER.warning(
                        "ZOE proactive voice "
                        "could not be delivered."
                    )

        except Exception:

            LOGGER.exception(
                "Unhandled proactive Telegram "
                "event error."
            )

        finally:

            try:

                event_queue.task_done()

            except Exception:

                pass

    LOGGER.info(
        "ZOE -> Telegram proactive "
        "event worker stopped."
    )


def start_proactive_notifications() -> bool:

    global _proactive_thread
    global _proactive_subscriber_id
    global _proactive_queue

    if not TELEGRAM_ENABLED:
        return False

    if TELEGRAM_ALLOWED_USER_ID is None:
        return False

    with _proactive_state_lock:

        if (
            _proactive_thread is not None
            and _proactive_thread.is_alive()
        ):

            return True

        try:

            (
                subscriber_id,
                subscriber_queue,
            ) = subscribe_runtime_events()

        except Exception:

            LOGGER.exception(
                "Failed to subscribe to "
                "ZOE runtime events."
            )

            return False

        _proactive_subscriber_id = (
            subscriber_id
        )

        _proactive_queue = (
            subscriber_queue
        )

        _proactive_stop_requested.clear()

        _proactive_thread = threading.Thread(
            target=_proactive_event_worker,
            name="ZOE-Telegram-Proactive",
            daemon=True,
        )

        _proactive_thread.start()

    LOGGER.info(
        "Subscribed to ZOE runtime events "
        "| subscriber=%s",
        subscriber_id,
    )

    return True


def stop_proactive_notifications(
    timeout: float = 5.0,
) -> None:

    global _proactive_thread
    global _proactive_subscriber_id
    global _proactive_queue

    with _proactive_state_lock:

        thread = (
            _proactive_thread
        )

        subscriber_id = (
            _proactive_subscriber_id
        )

        _proactive_stop_requested.set()

    if subscriber_id:

        try:

            unsubscribe_runtime_events(
                subscriber_id
            )

        except Exception:

            LOGGER.exception(
                "Failed to unsubscribe "
                "from ZOE runtime events."
            )

    if (
        thread is not None
        and thread is not threading.current_thread()
        and thread.is_alive()
    ):

        thread.join(
            timeout=timeout
        )

        if thread.is_alive():

            LOGGER.warning(
                "Proactive Telegram worker "
                "did not stop within %.1fs.",
                timeout,
            )

    with _proactive_state_lock:

        _proactive_thread = None
        _proactive_subscriber_id = None
        _proactive_queue = None

    LOGGER.info(
        "ZOE proactive Telegram bridge stopped."
    )


def is_proactive_notifications_running() -> bool:

    with _proactive_state_lock:

        return bool(
            _proactive_thread
            and _proactive_thread.is_alive()
            and _proactive_subscriber_id
        )


# ============================================================
# PUBLIC NOTIFICATION API
# ============================================================

def notify_telegram(
    text: str,
) -> bool:

    return send_telegram_message(
        text
    )


def notify_telegram_voice(
    text: str,
) -> bool:

    return send_telegram_voice(
        text
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "ZOE TELEGRAM INTERFACE"
    )

    print("=" * 60)

    if not TELEGRAM_ENABLED:

        print(
            "Telegram is disabled."
        )

        print(
            "Set TELEGRAM_ENABLED=true "
            "in .env"
        )

        raise SystemExit(0)

    if not TELEGRAM_BOT_TOKEN:

        print(
            "Missing TELEGRAM_BOT_TOKEN."
        )

        raise SystemExit(1)

    if TELEGRAM_ALLOWED_USER_ID is None:

        print(
            "Missing/invalid "
            "TELEGRAM_ALLOWED_USER_ID."
        )

        raise SystemExit(1)

    if (
        TELEGRAM_VOICE_ENABLED
        and shutil.which(
            TELEGRAM_VOICE_FFMPEG
        ) is None
    ):

        print(
            "FFmpeg is required for Telegram "
            "voice messages."
        )

        raise SystemExit(1)

    if not start_telegram_bot():

        print(
            "Failed to start Telegram bot."
        )

        raise SystemExit(1)

    print()

    print(
        "Telegram bot is running."
    )

    print(
        "Proactive notifications: enabled."
    )

    print(
        "Final-response waiting: enabled."
    )

    print(
        "Telegram voice messages: "
        f"{'enabled' if TELEGRAM_VOICE_ENABLED else 'disabled'}"
    )

    print(
        f"Response timeout: "
        f"{TELEGRAM_RESPONSE_TIMEOUT:.0f}s"
    )

    print(
        "Press Ctrl+C to stop."
    )

    try:

        while True:

            threading.Event().wait(
                1
            )

    except KeyboardInterrupt:

        print()
        print("Stopping...")

    finally:

        stop_telegram_bot()