from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import argparse
import atexit
import logging
import os
import queue
import re
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from pocket_tts import TTSModel


# ============================================================================
# CONFIG
# ============================================================================

VOICE = os.getenv("ZOE_TTS_VOICE", "eve").strip() or "eve"
LANGUAGE = os.getenv("ZOE_TTS_LANG", "english").strip() or "english"

# 1 is fastest and normally what ZOE wants.
SAMPLER_DECODE_STEPS = max(
    1,
    int(os.getenv("ZOE_TTS_STEPS", "1")),
)

# Larger chunks reduce Pocket TTS call overhead.
# 260 was reasonable, but 320 gives better throughput while still
# preserving streaming responsiveness.
MAX_CHUNK_LENGTH = max(
    120,
    int(os.getenv("ZOE_TTS_CHUNK", "320")),
)

# Keep this small. A huge audio queue increases latency when the user
# interrupts ZOE.
MAX_AUDIO_QUEUE = max(
    2,
    int(os.getenv("ZOE_TTS_QUEUE", "6")),
)

SAMPLE_RATE = 24_000

# Generation queue normally contains only the current utterance chunks.
MAX_GENERATION_QUEUE = max(
    8,
    MAX_AUDIO_QUEUE * 2,
)

TTS_DEVICE = (
    os.getenv("ZOE_TTS_DEVICE", "").strip()
    or None
)

AUDIO_BLOCKSIZE = int(
    os.getenv("ZOE_TTS_BLOCKSIZE", "0")
)

WORKER_POLL = 0.05
QUEUE_WAIT = 0.01


# ============================================================================
# LOGGING
# ============================================================================

LOGGER = logging.getLogger("zoe.tts")

if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[ZOE TTS] %(levelname)s: %(message)s"
        )
    )
    LOGGER.addHandler(handler)

LOGGER.setLevel(
    os.getenv(
        "ZOE_TTS_LOG_LEVEL",
        "INFO",
    ).upper()
)

LOGGER.propagate = False

# Keep noisy dependencies quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)


# ============================================================================
# DATA
# ============================================================================

@dataclass(frozen=True, slots=True)
class SpeechRequest:
    generation_id: int
    sequence: int
    text: str


@dataclass(slots=True)
class GenerationState:
    generation_id: int
    total_chunks: int

    completed_chunks: int = 0
    audio_chunks: int = 0

    generation_finished: bool = False
    generation_failed: bool = False

    playback_started: bool = False
    playback_finished: bool = False

    cancelled: bool = False
    error: str = ""

    speech_complete_event: threading.Event = field(
        default_factory=threading.Event
    )


# ============================================================================
# MODEL
# ============================================================================

_tts: TTSModel | None = None
_voice_state = None

_tts_lock = threading.RLock()


def preload() -> None:
    """Load Pocket TTS once."""

    global _tts
    global _voice_state

    if _tts is not None and _voice_state is not None:
        return

    with _tts_lock:

        if _tts is not None and _voice_state is not None:
            return

        LOGGER.info("Loading Pocket TTS...")

        model = TTSModel.load_model(
            language=LANGUAGE,
            sampler_decode_steps=SAMPLER_DECODE_STEPS,
        )

        voice_state = model.get_state_for_audio_prompt(
            VOICE
        )

        _tts = model
        _voice_state = voice_state

        LOGGER.info(
            "Pocket TTS ready: voice=%s language=%s steps=%s",
            VOICE,
            LANGUAGE,
            SAMPLER_DECODE_STEPS,
        )


# ============================================================================
# GENERATION IDS
# ============================================================================

_generation_id = 0
_generation_lock = threading.Lock()


def _next_generation_id() -> int:
    global _generation_id

    with _generation_lock:
        _generation_id += 1
        return _generation_id


def _current_generation_id() -> int:
    with _generation_lock:
        return _generation_id


def _is_current(generation_id: int) -> bool:
    with _generation_lock:
        return generation_id == _generation_id


# ============================================================================
# STATE
# ============================================================================

_states: dict[int, GenerationState] = {}
_states_lock = threading.RLock()


def _get_state(
    generation_id: int,
) -> GenerationState | None:

    with _states_lock:
        return _states.get(generation_id)


def _install_state(
    state: GenerationState,
) -> None:

    with _states_lock:
        _states.clear()
        _states[state.generation_id] = state


def _cancel_state(
    generation_id: int,
) -> None:

    state = _get_state(generation_id)

    if state is None:
        return

    with _states_lock:
        state.cancelled = True

    state.speech_complete_event.set()


# ============================================================================
# QUEUES
# ============================================================================

_generation_queue: queue.Queue[SpeechRequest] = queue.Queue(
    maxsize=MAX_GENERATION_QUEUE
)

# Store FLOAT32 directly.
#
# Old implementation:
#
# Pocket TTS float
#       ↓
# int16
#       ↓
# float32
#       ↓
# sounddevice
#
# New implementation:
#
# Pocket TTS float
#       ↓
# float32
#       ↓
# sounddevice
#
# Less CPU and less copying.
_audio_queue: queue.Queue[
    tuple[int, np.ndarray]
] = queue.Queue(
    maxsize=MAX_AUDIO_QUEUE
)


def _clear_generation_queue() -> None:

    while True:
        try:
            _generation_queue.get_nowait()
        except queue.Empty:
            break
        else:
            _generation_queue.task_done()


def _clear_audio_queue() -> None:

    while True:
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            break
        else:
            _audio_queue.task_done()


# ============================================================================
# AUDIO OUTPUT
# ============================================================================

_audio_stream: sd.OutputStream | None = None
_audio_stream_lock = threading.RLock()


def _get_audio_stream() -> sd.OutputStream:

    global _audio_stream

    with _audio_stream_lock:

        if _audio_stream is not None:

            try:
                if _audio_stream.active:
                    return _audio_stream
            except Exception:
                pass

            try:
                _audio_stream.close()
            except Exception:
                pass

            _audio_stream = None

        LOGGER.debug(
            "Opening audio device=%s rate=%s",
            TTS_DEVICE or "default",
            SAMPLE_RATE,
        )

        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=TTS_DEVICE,
            blocksize=AUDIO_BLOCKSIZE,
            # Do not force ultra-low latency.
            # Windows WASAPI/PortAudio can become less stable.
            latency="high",
        )

        stream.start()

        _audio_stream = stream

        LOGGER.info(
            "Audio output ready: %s",
            TTS_DEVICE or "Windows default",
        )

        return stream


def _stop_audio_stream() -> None:

    global _audio_stream

    with _audio_stream_lock:

        stream = _audio_stream
        _audio_stream = None

        if stream is None:
            return

        try:
            stream.abort()
        except Exception:
            pass

        try:
            stream.close()
        except Exception:
            pass


def _play_audio(
    audio: np.ndarray,
) -> None:

    if audio.size == 0:
        return

    stream = _get_audio_stream()

    stream.write(
        audio.reshape(-1, 1)
    )


# ============================================================================
# SPEAKING STATE
# ============================================================================

_speaking = False
_speaking_lock = threading.Lock()


def is_speaking() -> bool:

    with _speaking_lock:
        return _speaking


def _set_speaking(
    value: bool,
) -> None:

    global _speaking

    with _speaking_lock:
        _speaking = bool(value)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

# IMPORTANT:
#
# This normalizer is intentionally conservative.
#
# Pocket TTS is an English TTS model, not a code reader.
# Trying to convert every number, symbol, URL and date manually causes
# more pronunciation problems than it solves.
#
# We only fix patterns that are consistently problematic.


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _clean_markdown(text: str) -> str:

    # Remove fenced code blocks completely.
    text = re.sub(
        r"```[\s\S]*?```",
        "",
        text,
    )

    # Markdown links -> visible text.
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Inline code.
    text = re.sub(
        r"`([^`]+)`",
        r"\1",
        text,
    )

    # Headers.
    text = re.sub(
        r"(?m)^\s*#{1,6}\s+",
        "",
        text,
    )

    # Bullets.
    text = re.sub(
        r"(?m)^\s*[-*•]\s+",
        "",
        text,
    )

    # Bold / italic markers.
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    text = text.replace("_", " ")

    return text


# ---------------------------------------------------------------------------
# Security codes
# ---------------------------------------------------------------------------

_SECURITY_CODE = re.compile(
    r"\b("
    r"security\s+code|"
    r"verification\s+code|"
    r"verify\s+code|"
    r"authentication\s+code|"
    r"auth\s+code|"
    r"access\s+code|"
    r"one[- ]time\s+password|"
    r"one[- ]time\s+code|"
    r"otp"
    r")\s*[:#-]?\s*(\d{4,8})\b",
    re.IGNORECASE,
)


_DIGITS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _replace_security_codes(
    text: str,
) -> str:

    def replace(match: re.Match[str]) -> str:

        label = match.group(1)
        code = match.group(2)

        spoken = " ".join(
            _DIGITS[d]
            for d in code
        )

        return f"{label} {spoken}"

    return _SECURITY_CODE.sub(
        replace,
        text,
    )


# ---------------------------------------------------------------------------
# Tech model names
# ---------------------------------------------------------------------------

# Only split letter -> digit boundaries.
#
# T480    -> T 480
# RTX4090 -> RTX 4090
# USB3    -> USB 3
# A320    -> A 320
#
# Do NOT touch:
#
# 5G
# 2FA
# 802.11
# 192.168.1.1
#
_LETTER_DIGIT = re.compile(
    r"(?<=[A-Za-z])(?=\d)"
)


def _split_tech_names(
    text: str,
) -> str:

    return _LETTER_DIGIT.sub(
        " ",
        text,
    )


# ---------------------------------------------------------------------------
# Common acronyms
# ---------------------------------------------------------------------------

_ACRONYMS = {
    "ZOE": "Zoe",
}


_ACRONYM_PATTERN = re.compile(
    r"\b("
    + "|".join(
        re.escape(x)
        for x in sorted(
            _ACRONYMS,
            key=len,
            reverse=True,
        )
    )
    + r")\b",
    re.IGNORECASE,
)


def _replace_acronyms(
    text: str,
) -> str:

    def replace(match: re.Match[str]) -> str:

        value = match.group(1)

        return _ACRONYMS.get(
            value.upper(),
            value,
        )

    return _ACRONYM_PATTERN.sub(
        replace,
        text,
    )


# ---------------------------------------------------------------------------
# Times
# ---------------------------------------------------------------------------

_TIME_12 = re.compile(
    r"\b(0?\d|1[0-2]):([0-5]\d)\s*(AM|PM)\b",
    re.IGNORECASE,
)


def _replace_times(
    text: str,
) -> str:

    def replace(match: re.Match[str]) -> str:

        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3).upper()

        period_text = (
            "A M"
            if period == "AM"
            else "P M"
        )

        if minute == 0:
            return f"{hour} {period_text}"

        return f"{hour} {minute} {period_text}"

    return _TIME_12.sub(
        replace,
        text,
    )


# ---------------------------------------------------------------------------
# Percentages
# ---------------------------------------------------------------------------

_PERCENT = re.compile(
    r"\b(\d+(?:\.\d+)?)%",
)


def _replace_percentages(
    text: str,
) -> str:

    return _PERCENT.sub(
        lambda m: f"{m.group(1)} percent",
        text,
    )


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

_EMAIL = re.compile(
    r"\b"
    r"[A-Za-z0-9._%+-]+"
    r"@"
    r"[A-Za-z0-9.-]+"
    r"\."
    r"[A-Za-z]{2,}"
    r"\b",
)


def _replace_emails(
    text: str,
) -> str:

    def replace(match: re.Match[str]) -> str:

        value = match.group(0)

        return (
            value
            .replace("@", " at ")
            .replace(".", " dot ")
            .replace("_", " underscore ")
            .replace("-", " dash ")
        )

    return _EMAIL.sub(
        replace,
        text,
    )


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

_URL = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)


def _replace_urls(
    text: str,
) -> str:

    def replace(match: re.Match[str]) -> str:

        value = match.group(0)

        trailing = ""

        while value and value[-1] in ".,!?;:)":

            trailing = value[-1] + trailing
            value = value[:-1]

        value = re.sub(
            r"^https?://",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = re.sub(
            r"^www\.",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = (
            value
            .replace("/", " slash ")
            .replace("_", " underscore ")
            .replace("-", " dash ")
            .replace(".", " dot ")
        )

        return value + trailing

    return _URL.sub(
        replace,
        text,
    )


# ---------------------------------------------------------------------------
# Decimal numbers
#
# Do NOT convert all integers to words.
#
# Pocket TTS is much better at:
#
# "I have 5 tasks."
#
# than:
#
# "I have five tasks."
#
# So ordinary numbers are left alone.
# ---------------------------------------------------------------------------

_DECIMAL = re.compile(
    r"(?<![\w.])-?\d+\.\d+(?![\w.])"
)


def _replace_decimal_numbers(
    text: str,
) -> str:

    def replace(match: re.Match[str]) -> str:

        value = match.group(0)

        if value.startswith("-"):
            value = "minus " + value[1:]

        return value

    return _DECIMAL.sub(
        replace,
        text,
    )


# ---------------------------------------------------------------------------
# MASTER NORMALIZER
# ---------------------------------------------------------------------------

def _normalize_speech(
    text: str,
) -> str:

    text = str(
        text or ""
    ).strip()

    if not text:
        return ""

    text = _clean_markdown(text)

    # Structured things first.
    text = _replace_emails(text)
    text = _replace_urls(text)
    text = _replace_security_codes(text)

    # Tech names.
    text = _split_tech_names(text)

    # Only known problematic acronym pronunciations.
    text = _replace_acronyms(text)

    # Small targeted fixes.
    text = _replace_times(text)
    text = _replace_percentages(text)
    text = _replace_decimal_numbers(text)

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================================
# CHUNKING
# ============================================================================

# Split only when there is actual sentence punctuation.
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+"
)


def _split_long_text(
    text: str,
) -> list[str]:

    words = text.split()

    if not words:
        return []

    result: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:

        additional = (
            len(word)
            + (1 if current else 0)
        )

        if (
            current
            and length + additional
            > MAX_CHUNK_LENGTH
        ):

            result.append(
                " ".join(current)
            )

            current = [word]
            length = len(word)

        else:

            current.append(word)
            length += additional

    if current:
        result.append(
            " ".join(current)
        )

    return result


def _split_sentences(
    text: str,
) -> list[str]:

    text = _normalize_speech(text)

    if not text:
        return []

    sentences = _SENTENCE_SPLIT.split(text)

    chunks: list[str] = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) <= MAX_CHUNK_LENGTH:
            chunks.append(sentence)
        else:
            chunks.extend(
                _split_long_text(sentence)
            )

    return chunks


# ============================================================================
# AUDIO CONVERSION
# ============================================================================

def _audio_to_float32(
    audio: object,
) -> np.ndarray:

    samples = np.asarray(
        audio,
        dtype=np.float32,
    )

    if samples.size == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    # Pocket TTS should already return normalized audio.
    # Only clip when necessary.
    if (
        samples.min() < -1.0
        or samples.max() > 1.0
    ):
        samples = np.clip(
            samples,
            -1.0,
            1.0,
        )

    return np.ascontiguousarray(
        samples,
        dtype=np.float32,
    )


# ============================================================================
# COMPLETION
# ============================================================================

def _fail_generation(
    generation_id: int,
    error: Exception | str,
) -> None:

    state = _get_state(
        generation_id
    )

    if state is None:
        return

    with _states_lock:

        state.generation_failed = True
        state.generation_finished = True
        state.error = str(error)

    state.speech_complete_event.set()

    if _is_current(generation_id):
        _set_speaking(False)

    LOGGER.error(
        "Generation failed (%s): %s",
        generation_id,
        error,
    )


def _complete_generation(
    generation_id: int,
) -> None:

    state = _get_state(
        generation_id
    )

    if state is None:
        return

    with _states_lock:

        if state.playback_finished:
            return

        state.playback_finished = True

    state.speech_complete_event.set()

    if _is_current(generation_id):
        _set_speaking(False)


# ============================================================================
# WORKERS
# ============================================================================

_shutdown_event = threading.Event()

_workers_started = False
_workers_lock = threading.RLock()

_generation_worker: threading.Thread | None = None
_playback_worker: threading.Thread | None = None


# ============================================================================
# GENERATION WORKER
# ============================================================================

def _generation_worker_loop() -> None:

    while not _shutdown_event.is_set():

        try:
            request = _generation_queue.get(
                timeout=WORKER_POLL
            )
        except queue.Empty:
            continue

        try:

            generation_id = request.generation_id

            if not _is_current(generation_id):
                continue

            state = _get_state(generation_id)

            if (
                state is None
                or state.cancelled
            ):
                continue

            model = _tts
            voice_state = _voice_state

            if model is None or voice_state is None:
                raise RuntimeError(
                    "Pocket TTS is not loaded."
                )

            # Generate one chunk.
            #
            # The playback worker is consuming the previous generated
            # audio at the same time, so generation and playback overlap.
            stream = model.generate_audio_stream(
                model_state=voice_state,
                text_to_generate=request.text,
                copy_state=True,
            )

            for audio in stream:

                if _shutdown_event.is_set():
                    break

                if not _is_current(generation_id):
                    break

                state = _get_state(
                    generation_id
                )

                if (
                    state is None
                    or state.cancelled
                ):
                    break

                samples = _audio_to_float32(audio)

                if samples.size == 0:
                    continue

                # Backpressure.
                while not _shutdown_event.is_set():

                    if not _is_current(generation_id):
                        break

                    state = _get_state(
                        generation_id
                    )

                    if (
                        state is None
                        or state.cancelled
                    ):
                        break

                    try:

                        _audio_queue.put(
                            (
                                generation_id,
                                samples,
                            ),
                            timeout=QUEUE_WAIT,
                        )

                        state.audio_chunks += 1
                        break

                    except queue.Full:
                        continue

            state = _get_state(
                generation_id
            )

            if (
                state is not None
                and _is_current(generation_id)
                and not state.cancelled
            ):

                state.completed_chunks += 1

                if (
                    state.completed_chunks
                    >= state.total_chunks
                ):

                    state.generation_finished = True

        except Exception as exc:

            if _is_current(
                request.generation_id
            ):
                _fail_generation(
                    request.generation_id,
                    exc,
                )

        finally:

            _generation_queue.task_done()


# ============================================================================
# PLAYBACK WORKER
# ============================================================================

def _playback_worker_loop() -> None:

    active_generation: int | None = None

    while not _shutdown_event.is_set():

        try:

            generation_id, audio = (
                _audio_queue.get(
                    timeout=WORKER_POLL
                )
            )

        except queue.Empty:

            current_id = (
                _current_generation_id()
            )

            state = _get_state(
                current_id
            )

            if (
                state is not None
                and state.generation_finished
                and not state.generation_failed
                and not state.cancelled
                and state.audio_chunks > 0
                and _audio_queue.empty()
            ):

                if active_generation == current_id:

                    _complete_generation(
                        current_id
                    )

                    active_generation = None

            continue

        try:

            if not _is_current(
                generation_id
            ):
                continue

            state = _get_state(
                generation_id
            )

            if (
                state is None
                or state.cancelled
                or state.generation_failed
            ):
                continue

            if active_generation != generation_id:

                _get_audio_stream()

                active_generation = generation_id

                state.playback_started = True

            _play_audio(audio)

        except (
            sd.PortAudioError,
            BrokenPipeError,
            OSError,
            ValueError,
        ) as exc:

            if _is_current(
                generation_id
            ):
                _fail_generation(
                    generation_id,
                    exc,
                )

            _stop_audio_stream()

            active_generation = None

        except Exception as exc:

            LOGGER.exception(
                "Unexpected playback error."
            )

            if _is_current(
                generation_id
            ):
                _fail_generation(
                    generation_id,
                    exc,
                )

            _stop_audio_stream()

            active_generation = None

        finally:

            _audio_queue.task_done()

    _stop_audio_stream()


# ============================================================================
# WORKER MANAGEMENT
# ============================================================================

def _ensure_workers() -> None:

    global _workers_started
    global _generation_worker
    global _playback_worker

    with _workers_lock:

        if (
            _workers_started
            and _generation_worker is not None
            and _generation_worker.is_alive()
            and _playback_worker is not None
            and _playback_worker.is_alive()
        ):
            return

        _shutdown_event.clear()

        _generation_worker = threading.Thread(
            target=_generation_worker_loop,
            name="ZOE-TTS-Generator",
            daemon=True,
        )

        _playback_worker = threading.Thread(
            target=_playback_worker_loop,
            name="ZOE-TTS-Playback",
            daemon=True,
        )

        _generation_worker.start()
        _playback_worker.start()

        _workers_started = True


# ============================================================================
# SPEAK
# ============================================================================

_speak_lock = threading.RLock()


def speak_async(
    text: str,
    on_complete: Callable[[], None] | None = None,
) -> int | None:
    """
    Start speaking immediately.

    New speech automatically replaces the current speech.
    """

    text = str(
        text or ""
    ).strip()

    if not text:
        return None

    chunks = _split_sentences(text)

    if not chunks:
        return None

    preload()
    _ensure_workers()

    with _speak_lock:

        old_generation = (
            _current_generation_id()
        )

        generation_id = (
            _next_generation_id()
        )

        # Kill old generation first.
        if old_generation:
            _cancel_state(
                old_generation
            )

        # Drop anything waiting from the old utterance.
        _clear_generation_queue()
        _clear_audio_queue()

        # Stop audible audio immediately.
        _stop_audio_stream()

        state = GenerationState(
            generation_id=generation_id,
            total_chunks=len(chunks),
        )

        _install_state(state)

        _set_speaking(True)

        for sequence, chunk in enumerate(chunks):

            try:

                _generation_queue.put_nowait(
                    SpeechRequest(
                        generation_id=generation_id,
                        sequence=sequence,
                        text=chunk,
                    )
                )

            except queue.Full:

                state.cancelled = True
                state.generation_failed = True
                state.generation_finished = True
                state.error = (
                    "TTS generation queue is full."
                )

                state.speech_complete_event.set()

                _set_speaking(False)

                _clear_generation_queue()

                return None

    if on_complete is not None:

        def completion_waiter() -> None:

            state.speech_complete_event.wait()

            if not _is_current(
                generation_id
            ):
                return

            if (
                state.cancelled
                or state.generation_failed
                or not state.playback_finished
            ):
                return

            try:
                on_complete()
            except Exception:
                LOGGER.exception(
                    "TTS completion callback failed."
                )

        threading.Thread(
            target=completion_waiter,
            name=f"ZOE-TTS-Completion-{generation_id}",
            daemon=True,
        ).start()

    return generation_id


def speak(
    text: str,
    on_complete: Callable[[], None] | None = None,
) -> bool:
    """
    Blocking speech.

    Returns True only if the utterance fully played.
    """

    generation_id = speak_async(
        text,
        on_complete=on_complete,
    )

    if generation_id is None:
        return False

    state = _get_state(
        generation_id
    )

    if state is None:
        return False

    state.speech_complete_event.wait()

    return (
        _is_current(generation_id)
        and state.playback_finished
        and not state.cancelled
        and not state.generation_failed
    )


# ============================================================================
# STOP
# ============================================================================

def stop() -> None:
    """
    Immediately stop generation and playback.
    """

    with _speak_lock:

        current = (
            _current_generation_id()
        )

        # Invalidate current work immediately.
        _next_generation_id()

        if current:
            _cancel_state(current)

        _set_speaking(False)

        _clear_generation_queue()
        _clear_audio_queue()

        _stop_audio_stream()


# ============================================================================
# SHUTDOWN
# ============================================================================

def shutdown() -> None:

    global _workers_started
    global _generation_worker
    global _playback_worker

    with _speak_lock:

        current = (
            _current_generation_id()
        )

        _next_generation_id()

        if current:
            _cancel_state(current)

        _set_speaking(False)

        _clear_generation_queue()
        _clear_audio_queue()

        _shutdown_event.set()

        _stop_audio_stream()

    current_thread = threading.current_thread()

    with _workers_lock:

        workers = (
            _generation_worker,
            _playback_worker,
        )

    for worker in workers:

        if (
            worker is not None
            and worker is not current_thread
        ):

            worker.join(
                timeout=1.0
            )

    with _workers_lock:

        _generation_worker = None
        _playback_worker = None
        _workers_started = False


# ============================================================================
# STATUS
# ============================================================================

def get_status() -> dict[str, object]:

    generation_id = (
        _current_generation_id()
    )

    state = _get_state(
        generation_id
    )

    with _audio_stream_lock:

        stream_alive = False

        if _audio_stream is not None:

            try:
                stream_alive = bool(
                    _audio_stream.active
                )
            except Exception:
                stream_alive = False

    return {
        "speaking": is_speaking(),

        "generation_id": generation_id,

        "generation_queue": (
            _generation_queue.qsize()
        ),

        "audio_queue": (
            _audio_queue.qsize()
        ),

        # Compatibility with older ZOE backend.
        "audio_process_alive": stream_alive,

        "audio_stream_alive": stream_alive,

        "model_loaded": (
            _tts is not None
            and _voice_state is not None
        ),

        "generation_finished": (
            state.generation_finished
            if state
            else False
        ),

        "generation_failed": (
            state.generation_failed
            if state
            else False
        ),

        "playback_started": (
            state.playback_started
            if state
            else False
        ),

        "playback_finished": (
            state.playback_finished
            if state
            else False
        ),

        "error": (
            state.error
            if state
            else ""
        ),
    }


# ============================================================================
# VOICE FILE GENERATION
# ============================================================================

def generate_voice_file(
    text: str,
    output_path: str | None = None,
) -> str | None:
    """
    Generate speech to a WAV file.

    This does NOT play the audio.
    """

    text = str(
        text or ""
    ).strip()

    if not text:
        return None

    chunks = _split_sentences(text)

    if not chunks:
        return None

    preload()

    model = _tts
    voice_state = _voice_state

    if model is None or voice_state is None:
        LOGGER.error(
            "Pocket TTS is not initialized."
        )
        return None

    owns_file = output_path is None

    if output_path is None:

        fd, output_path = tempfile.mkstemp(
            prefix="zoe_voice_",
            suffix=".wav",
        )

        os.close(fd)

    try:

        audio_parts: list[np.ndarray] = []

        for chunk in chunks:

            stream = model.generate_audio_stream(
                model_state=voice_state,
                text_to_generate=chunk,
                copy_state=True,
            )

            for audio in stream:

                samples = _audio_to_float32(audio)

                if samples.size:
                    audio_parts.append(samples)

        if not audio_parts:

            LOGGER.warning(
                "Pocket TTS produced no audio."
            )

            if owns_file:

                try:
                    os.remove(output_path)
                except OSError:
                    pass

            return None

        samples = np.concatenate(
            audio_parts
        )

        # WAV requires integer PCM.
        pcm = np.clip(
            samples,
            -1.0,
            1.0,
        )

        pcm = (
            pcm * 32767.0
        ).astype(
            np.int16
        )

        with wave.open(
            output_path,
            "wb",
        ) as wav:

            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(
                pcm.tobytes()
            )

        return output_path

    except Exception:

        LOGGER.exception(
            "Failed to generate voice file."
        )

        if owns_file:

            try:
                os.remove(output_path)
            except OSError:
                pass

        return None


# ============================================================================
# TESTS
# ============================================================================

def _test_normalization() -> None:

    tests = [
        "ThinkPad T480",
        "RTX4090 GPU",
        "iPhone15",
        "USB3",
        "5G",
        "2FA",
        "CPU and GPU",
        "27 degrees Celsius",
        "99.5%",
        "3.14",
        "https://example.com/api/v1",
        "zoe@example.com",
        "security code 4413796",
        "10:30 PM",
        "192.168.0.101",
    ]

    print()
    print("=" * 72)
    print("ZOE TTS NORMALIZATION")
    print("=" * 72)

    for value in tests:

        print()
        print("INPUT : ", value)
        print(
            "OUTPUT:",
            _normalize_speech(value),
        )

    print()
    print("=" * 72)


def _test() -> None:

    text = (
        "Hello Sir. "
        "This is the ZOE Pocket TTS production test. "
        "The CPU and GPU should sound natural. "
        "The ThinkPad T480 is being tested with an RTX4090 GPU. "
        "The iPhone15 is another example. "
        "USB3 is also being tested. "
        "The current temperature is 27 degrees Celsius. "
        "The server is available at "
        "https://example.com/api/v1. "
        "The email is zoe@example.com. "
        "You received a security code 4413796. "
        "The code should be spoken digit by digit. "
        "Playback should finish naturally."
    )

    LOGGER.info(
        "TTS test starting..."
    )

    LOGGER.info(
        "Normalized: %s",
        _normalize_speech(text),
    )

    started = time.monotonic()

    success = speak(text)

    LOGGER.info(
        "TTS test finished: success=%s elapsed=%.2fs",
        success,
        time.monotonic() - started,
    )


# ============================================================================
# CLI
# ============================================================================

def _main() -> None:

    parser = argparse.ArgumentParser(
        description="ZOE Pocket TTS"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run TTS playback test.",
    )

    parser.add_argument(
        "--test-normalization",
        action="store_true",
        help="Run normalization test.",
    )

    args = parser.parse_args()

    if args.test_normalization:
        _test_normalization()
        return

    _test()


# ============================================================================
# CLEANUP
# ============================================================================

atexit.register(shutdown)


if __name__ == "__main__":

    try:
        _main()

    except KeyboardInterrupt:

        LOGGER.info(
            "Interrupted."
        )

    finally:

        shutdown()