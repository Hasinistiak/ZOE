from __future__ import annotations

import asyncio
import logging
import os
import queue
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


# Telegram's maximum outgoing message size.
TELEGRAM_MAX_MESSAGE_LENGTH = 4096


# Maximum incoming message accepted by ZOE.
TELEGRAM_MAX_INPUT_LENGTH = 12000


# ------------------------------------------------------------
# Maximum time Telegram will wait for the REAL final ZOE
# response.
#
# This is especially important for agent requests.
#
# Example:
#
#   Telegram -> ZOE
#             -> agent starts
#             -> research/tool work
#             -> synthesis
#             -> final answer
#             -> Telegram
#
# It does NOT return the initial agent acknowledgement.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Runtime events which should become unsolicited Telegram
# notifications.
#
# "response" is deliberately NOT included.
#
# Normal responses are already sent by _message_handler().
#
# The runtime uses:
#
#   output + type=notification
#   output + type=reminder
#
# for proactive output.
# ------------------------------------------------------------

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
    """
    Only the configured Telegram numeric user ID
    may control ZOE.
    """

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
            "Normal text:\n"
            "  Send any message to ZOE.\n\n"
            "Agent requests:\n"
            "  ZOE waits for the final synthesized "
            "answer before replying.\n\n"
            "Proactive:\n"
            "  ZOE can send reminders and "
            "status notifications automatically.\n\n"
            "Commands:\n"
            "  /start   Start the interface\n"
            "  /help    Show this help\n"
            "  /status  Show ZOE status\n"
            "  /mute    Mute ZOE voice output\n"
            "  /unmute  Enable ZOE voice output\n"
            "  /stop    Stop current ZOE speech\n\n"
            "Telegram responses do not trigger TTS."
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
            f"{'YES' if proactive_running else 'NO'}"
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
            (
                "I couldn't retrieve the "
                "current ZOE status."
            ),
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
            (
                "I couldn't stop the "
                "current speech."
            ),
        )


# ============================================================
# NORMAL TELEGRAM MESSAGE
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

    # --------------------------------------------------------
    # Protect ZOE from unnecessarily large Telegram payloads.
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # run_zoe is synchronous.
        #
        # Never block Telegram's asyncio event loop.
        #
        # Arguments:
        #
        #   text
        #       User message.
        #
        #   False
        #       Telegram must not trigger TTS.
        #
        #   True
        #       Wait for the REAL final ZOE response.
        #
        #   TELEGRAM_RESPONSE_TIMEOUT
        #       Maximum time to wait.
        #
        # This is the important fix for agent requests.
        #
        # Previously:
        #
        #   Telegram -> run_zoe()
        #            -> agent acknowledgement
        #            -> Telegram replies immediately
        #
        # Now:
        #
        #   Telegram -> run_zoe()
        #            -> agent starts
        #            -> agent completes
        #            -> final synthesis
        #            -> runtime resolves interaction
        #            -> Telegram replies
        # ----------------------------------------------------

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

            await _reply(
                update,
                response_text,
            )

            return

        # ----------------------------------------------------
        # No usable final response.
        #
        # This should be unusual because the runtime's
        # completion mechanism should either return the final
        # answer or raise an error.
        # ----------------------------------------------------

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
# APPLICATION CONSTRUCTION
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

        # ----------------------------------------------------
        # Start the runtime event subscriber AFTER Telegram's
        # bot loop is running.
        #
        # This allows proactive events to be forwarded safely
        # through the same asyncio loop.
        # ----------------------------------------------------

        start_proactive_notifications()

        _bot_started.set()

        LOGGER.info(
            "Telegram bot polling started."
        )

        LOGGER.info(
            "Telegram proactive notification "
            "bridge started."
        )

        while not _bot_stop_requested.is_set():

            await asyncio.sleep(
                0.5
            )

    finally:

        LOGGER.info(
            "Stopping Telegram bot..."
        )

        # ----------------------------------------------------
        # Stop proactive event bridge first.
        # ----------------------------------------------------

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

    loop = (
        asyncio.new_event_loop()
    )

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
# TELEGRAM OUTBOUND
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
                chat_id=(
                    TELEGRAM_ALLOWED_USER_ID
                ),
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
    """
    Thread-safe synchronous interface for ZOE.

    Can safely be called from:
      - Runtime threads
      - Agent threads
      - Status threads
      - Flask
      - CLI
      - Other background workers
    """

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
# PROACTIVE ZOE -> TELEGRAM
# ============================================================

def _extract_proactive_text(
    event: object,
) -> Optional[str]:
    """
    Extract only genuine proactive output events.

    Runtime output looks like:

        {
            "type": "output",
            "data": {
                "type": "notification",
                "text": "...",
                ...
            }
        }

    or:

        {
            "type": "output",
            "data": {
                "type": "reminder",
                "text": "...",
                ...
            }
        }

    Normal responses use:

        data["type"] == "response"

    and are intentionally ignored here because the
    Telegram message handler sends those itself.
    """

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

            success = send_telegram_message(
                text
            )

            if success:

                LOGGER.info(
                    "ZOE proactive notification "
                    "delivered to Telegram."
                )

            else:

                LOGGER.warning(
                    "ZOE proactive notification "
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

        # ----------------------------------------------------
        # Subscribe to the central ZOE runtime event bus.
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Remove subscription first.
    # --------------------------------------------------------

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
# SIMPLE NOTIFICATION ALIAS
# ============================================================

def notify_telegram(
    text: str,
) -> bool:
    """
    Public notification API for other ZOE modules.

    Example:

        notify_telegram(
            "Sir, your reminder is due."
        )
    """

    return send_telegram_message(
        text
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ZOE TELEGRAM INTERFACE")
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
        print(
            "Stopping..."
        )

    finally:

        stop_telegram_bot()