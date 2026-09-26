from __future__ import annotations

"""
ZOE Pocket TTS
==============

Production-oriented, low-latency Pocket TTS service for Linux.

Design:
    text -> normalization/chunking
         -> generation worker
         -> bounded PCM queue
         -> playback worker
         -> persistent aplay process

Important properties:
    * New speech immediately supersedes old speech.
    * Generation and playback overlap.
    * The audio process is reused between utterances when possible.
    * Queue backpressure prevents unbounded RAM growth.
    * Cancellation is generation-token based, so stale audio cannot leak
      into a newer utterance.
    * Completion callbacks run only after real playback has drained.
    * First-use model loading is explicit through preload().
"""

from dataclasses import dataclass, field
from typing import Callable
import atexit
import logging
import os
import queue
import re
import subprocess
import threading
import time

import numpy as np
from pocket_tts import TTSModel


# ============================================================================
# CONFIG
# ============================================================================

VOICE = os.getenv("ZOE_TTS_VOICE", "eve").strip() or "eve"
LANGUAGE = os.getenv("ZOE_TTS_LANG", "english").strip() or "english"

SAMPLER_DECODE_STEPS = max(
    1,
    int(os.getenv("ZOE_TTS_STEPS", "1")),
)

# Smaller chunks reduce time-to-first-audio. Larger chunks reduce per-request
# Pocket TTS overhead. 260 is a reasonable low-latency default.
MAX_CHUNK_LENGTH = max(
    80,
    int(os.getenv("ZOE_TTS_CHUNK", "260")),
)

# Number of generated audio blocks allowed to wait for playback.
MAX_AUDIO_QUEUE = max(
    2,
    int(os.getenv("ZOE_TTS_QUEUE", "8")),
)

SAMPLE_RATE = 24_000

# Queue wait intervals are deliberately short so cancellation is responsive.
QUEUE_WAIT = 0.025
WORKER_POLL = 0.10

# Don't keep thousands of generation requests in memory.
MAX_GENERATION_QUEUE = max(
    16,
    MAX_AUDIO_QUEUE * 2,
)

# aplay is intentionally kept alive between normal utterances.
APLAY_COMMAND = (
    "aplay",
    "-q",
    "-t",
    "raw",
    "-f",
    "S16_LE",
    "-c",
    "1",
    "-r",
    str(SAMPLE_RATE),
    "-",
)

LOGGER = logging.getLogger("zoe.tts")

if not LOGGER.handlers:
    logging.basicConfig(
        level=os.getenv("ZOE_TTS_LOG_LEVEL", "INFO").upper(),
        format="[ZOE TTS] %(levelname)s: %(message)s",
    )


# ============================================================================
# DATA TYPES
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

    generated_chunks: int = 0
    audio_chunks_enqueued: int = 0

    generation_finished: bool = False
    generation_failed: bool = False

    playback_started: bool = False
    playback_finished: bool = False

    cancelled: bool = False
    error: str = ""

    speech_complete_event: threading.Event = field(
        default_factory=threading.Event,
    )


# ============================================================================
# MODEL STATE
# ============================================================================

_tts: TTSModel | None = None
_voice_state = None
_tts_init_lock = threading.RLock()


def preload() -> None:
    """
    Load Pocket TTS and the selected voice once.

    Call this during application startup if low first-utterance latency matters.
    Calling it from speak_async() is still safe.
    """
    global _tts, _voice_state

    with _tts_init_lock:
        if _tts is not None and _voice_state is not None:
            return

        LOGGER.info("Loading Pocket TTS...")

        model = TTSModel.load_model(
            language=LANGUAGE,
            sampler_decode_steps=SAMPLER_DECODE_STEPS,
        )
        voice_state = model.get_state_for_audio_prompt(VOICE)

        _tts = model
        _voice_state = voice_state

        LOGGER.info("Pocket TTS loaded.")


# ============================================================================
# GENERATION TOKEN
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


def _is_current_generation(generation_id: int) -> bool:
    with _generation_lock:
        return generation_id == _generation_id


# ============================================================================
# GENERATION STATE
# ============================================================================

_states: dict[int, GenerationState] = {}
_states_lock = threading.RLock()


def _get_state(generation_id: int) -> GenerationState | None:
    with _states_lock:
        return _states.get(generation_id)


def _cancel_state(generation_id: int) -> None:
    state = _get_state(generation_id)
    if state is None:
        return

    with _states_lock:
        state.cancelled = True
        state.speech_complete_event.set()


def _install_state(state: GenerationState) -> None:
    with _states_lock:
        _states.clear()
        _states[state.generation_id] = state


# ============================================================================
# QUEUES
# ============================================================================

_generation_queue: queue.Queue[SpeechRequest] = queue.Queue(
    maxsize=MAX_GENERATION_QUEUE,
)

_audio_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(
    maxsize=MAX_AUDIO_QUEUE,
)

_audio_event = threading.Event()


def _clear_generation_queue() -> None:
    while True:
        try:
            _generation_queue.get_nowait()
        except queue.Empty:
            return
        else:
            _generation_queue.task_done()


def _clear_audio_queue() -> None:
    while True:
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            _audio_event.clear()
            return
        else:
            _audio_queue.task_done()


# ============================================================================
# AUDIO PROCESS
# ============================================================================

_audio_process: subprocess.Popen[bytes] | None = None
_audio_process_lock = threading.RLock()


def _get_audio_process() -> subprocess.Popen[bytes]:
    """
    Return a live aplay process, creating one if necessary.

    The process stays open while ZOE is idle. This avoids paying process
    startup cost on every utterance.
    """
    global _audio_process

    with _audio_process_lock:
        if (
            _audio_process is not None
            and _audio_process.poll() is None
            and _audio_process.stdin is not None
        ):
            return _audio_process

        if _audio_process is not None:
            _terminate_process_locked(_audio_process)
            _audio_process = None

        process = subprocess.Popen(
            APLAY_COMMAND,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

        _audio_process = process
        return process


def _terminate_process_locked(
    process: subprocess.Popen[bytes],
) -> None:
    """Terminate a process while _audio_process_lock is held."""
    try:
        stdin = process.stdin
        if stdin is not None:
            try:
                stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

        if process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=0.35)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=0.35)
                except Exception:
                    pass
    except Exception as exc:
        LOGGER.debug("Audio process termination error: %s", exc)


def _stop_audio_process() -> None:
    global _audio_process

    with _audio_process_lock:
        process = _audio_process
        _audio_process = None

        if process is None:
            return

        _terminate_process_locked(process)


# ============================================================================
# SPEAKING STATE
# ============================================================================

_speaking = False
_speaking_lock = threading.Lock()


def is_speaking() -> bool:
    with _speaking_lock:
        return _speaking


def _set_speaking(value: bool) -> None:
    global _speaking

    with _speaking_lock:
        _speaking = bool(value)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

_ACRONYMS = {
    "AI": "A I",
    "API": "A P I",
    "CPU": "C P U",
    "GPU": "G P U",
    "RAM": "R A M",
    "ROM": "R O M",
    "SSD": "S S D",
    "HDD": "H D D",
    "USB": "U S B",
    "HDMI": "H D M I",
    "HTTP": "H T T P",
    "HTTPS": "H T T P S",
    "URL": "U R L",
    "UI": "U I",
    "UX": "U X",
    "OS": "O S",
    "PC": "P C",
    "FPS": "F P S",
    "DNS": "D N S",
    "IP": "I P",
    "TCP": "T C P",
    "UDP": "U D P",
    "SSH": "S S H",
    "LLM": "L L M",
    "STT": "S T T",
    "TTS": "T T S",
    "VAD": "V A D",
    "NLP": "N L P",
    "JSON": "J S O N",
    "XML": "X M L",
    "HTML": "H T M L",
    "CSS": "C S S",
    "JS": "J S",
    "JSX": "J S X",
    "CLI": "C L I",
    "GUI": "G U I",
    "WiFi": "Wi-Fi",
    "ZOE": "Zoe",
}

_ACRONYM_PATTERN = re.compile(
    r"\b("
    + "|".join(
        re.escape(key)
        for key in sorted(_ACRONYMS, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)

_DIGIT_WORDS = {
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

_NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)

_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

_SECURITY_PATTERN = re.compile(
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

_DECIMAL_PATTERN = re.compile(
    r"(?<![\w.])-?\d+\.\d+(?![\w.])",
)

_PERCENT_PATTERN = re.compile(
    r"(?<![\w.])-?\d+(?:\.\d+)?%",
)

_TEMPERATURE_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?)\s*"
    r"(C|F|Celsius|Fahrenheit)\b",
    re.IGNORECASE,
)

_TIME_12_PATTERN = re.compile(
    r"\b(0?\d|1[0-2]):([0-5]\d)\s*(AM|PM)\b",
    re.IGNORECASE,
)

_TIME_24_PATTERN = re.compile(
    r"\b([01]?\d|2[0-3]):([0-5]\d)\b",
)

_URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
)

_DATE_PATTERN = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December"
    r")\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,\s+|\s+)(\d{4})\b",
    re.IGNORECASE,
)

_INTEGER_PATTERN = re.compile(
    r"(?<![\w.])-?\d[\d,]*(?![\w.])",
)


def _number_to_words(number: int) -> str:
    if number < 0:
        return "minus " + _number_to_words(-number)

    if number < 20:
        return _NUMBER_WORDS[number]

    if number < 100:
        tens = (number // 10) * 10
        remainder = number % 10
        return _TENS[tens] + (
            f" {_NUMBER_WORDS[remainder]}" if remainder else ""
        )

    if number < 1000:
        hundreds = number // 100
        remainder = number % 100
        result = f"{_NUMBER_WORDS[hundreds]} hundred"
        return result + (
            f" {_number_to_words(remainder)}" if remainder else ""
        )

    for scale, name in (
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ):
        if number >= scale:
            major, remainder = divmod(number, scale)
            result = f"{_number_to_words(major)} {name}"
            return result + (
                f" {_number_to_words(remainder)}" if remainder else ""
            )

    return str(number)


def _clean_markdown(text: str) -> str:
    # Code blocks are usually implementation details and should not be read.
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Markdown links: keep visible label.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Inline code.
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Headings / bullets.
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = re.sub(r"(?m)^\s*_{3,}\s*$", "", text)

    # Common emphasis markers.
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    text = text.replace("_", " ")

    return text


def _replace_security_codes(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        code = match.group(2)
        spoken = " ".join(_DIGIT_WORDS[digit] for digit in code)
        return f"{label} {spoken}"

    return _SECURITY_PATTERN.sub(replace, text)


def _replace_decimals(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        negative = raw.startswith("-")
        raw = raw[1:] if negative else raw

        whole, decimal = raw.split(".", 1)

        result = (
            f"{_number_to_words(int(whole))} point "
            + " ".join(_DIGIT_WORDS[d] for d in decimal)
        )

        return "minus " + result if negative else result

    return _DECIMAL_PATTERN.sub(replace, text)


def _replace_percentages(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)[:-1]
        return f"{_replace_decimals(value)} percent" if "." in value else (
            f"{_number_to_words(int(value))} percent"
        )

    return _PERCENT_PATTERN.sub(replace, text)


def _replace_temperatures(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(1)
        unit = match.group(2).lower()

        unit_text = (
            "degrees Celsius"
            if unit in {"c", "celsius"}
            else "degrees Fahrenheit"
        )

        value_text = (
            _replace_decimals(value)
            if "." in value
            else _number_to_words(int(value))
        )

        return f"{value_text} {unit_text}"

    return _TEMPERATURE_PATTERN.sub(replace, text)


def _replace_times(text: str) -> str:
    def replace_12(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3).lower()

        if minute == 0:
            return f"{_number_to_words(hour)} {period}"

        # "05" should normally be spoken "five", not "zero five".
        return (
            f"{_number_to_words(hour)} "
            f"{_number_to_words(minute)} {period}"
        )

    text = _TIME_12_PATTERN.sub(replace_12, text)

    def replace_24(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))

        period = "PM" if hour >= 12 else "AM"
        display_hour = hour % 12 or 12

        if minute == 0:
            return f"{_number_to_words(display_hour)} {period}"

        return (
            f"{_number_to_words(display_hour)} "
            f"{_number_to_words(minute)} {period}"
        )

    return _TIME_24_PATTERN.sub(replace_24, text)


def _replace_urls(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)

        # Strip trailing punctuation that is sentence punctuation rather
        # than part of the URL.
        trailing = ""
        while value and value[-1] in ".,!?;:)":
            trailing = value[-1] + trailing
            value = value[:-1]

        value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^www\.", "", value, flags=re.IGNORECASE)

        value = value.replace("/", " slash ")
        value = value.replace("_", " underscore ")
        value = value.replace("-", " dash ")
        value = value.replace(".", " dot ")

        return value + trailing

    return _URL_PATTERN.sub(replace, text)


def _replace_emails(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        return (
            value.replace("@", " at ")
            .replace(".", " dot ")
            .replace("_", " underscore ")
            .replace("-", " dash ")
        )

    return _EMAIL_PATTERN.sub(replace, text)


def _replace_dates(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        month = match.group(1)
        day = int(match.group(2))
        year = int(match.group(3))

        return (
            f"{month} {_number_to_words(day)} "
            f"{_number_to_words(year)}"
        )

    return _DATE_PATTERN.sub(replace, text)


def _replace_acronyms(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        original = match.group(1)
        return _ACRONYMS.get(original.upper(), original)

    return _ACRONYM_PATTERN.sub(replace, text)


def _normalize_symbols(text: str) -> str:
    # Keep this conservative. Replacing every slash, dash, etc. makes normal
    # prose unnatural and is particularly harmful for code-like text.
    replacements = {
        "&": " and ",
        "+": " plus ",
        "=": " equals ",
        "#": " number ",
        "<": " less than ",
        ">": " greater than ",
    }

    for symbol, spoken in replacements.items():
        text = text.replace(symbol, spoken)

    return text


def _replace_integers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)

        try:
            number = int(raw.replace(",", ""))
        except ValueError:
            return raw

        return _number_to_words(number)

    return _INTEGER_PATTERN.sub(replace, text)


def _normalize_speech(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    text = _clean_markdown(text)
    text = _replace_emails(text)
    text = _replace_urls(text)
    text = _replace_security_codes(text)
    text = _replace_dates(text)
    text = _replace_times(text)
    text = _replace_temperatures(text)
    text = _replace_percentages(text)
    text = _replace_decimals(text)
    text = _replace_acronyms(text)
    text = _normalize_symbols(text)
    text = _replace_integers(text)

    # Normalize whitespace and excessive blank lines.
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================
# CHUNKING
# ============================================================================

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _split_long_segment(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        extra = len(word) + (1 if current else 0)

        if current and length + extra > MAX_CHUNK_LENGTH:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_speech(text)
    if not normalized:
        return []

    sentences = _SENTENCE_PATTERN.split(normalized)

    chunks: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= MAX_CHUNK_LENGTH:
            chunks.append(sentence)
        else:
            chunks.extend(_split_long_segment(sentence))

    return chunks


# ============================================================================
# PCM CONVERSION
# ============================================================================

def _audio_to_pcm(audio: object) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)

    if samples.size == 0:
        return np.empty(0, dtype=np.int16)

    # Avoid an expensive copy where possible. clip only when necessary.
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        samples = np.clip(samples, -1.0, 1.0)

    return np.multiply(
        samples,
        32767.0,
        dtype=np.float32,
    ).astype(np.int16, copy=False)


# ============================================================================
# GENERATION FAILURE / COMPLETION
# ============================================================================

def _fail_generation(
    generation_id: int,
    error: Exception | str,
) -> None:
    state = _get_state(generation_id)
    if state is None:
        return

    with _states_lock:
        state.generation_failed = True
        state.generation_finished = True
        state.error = str(error)

    state.speech_complete_event.set()

    LOGGER.error(
        "Generation failed (generation=%s): %s",
        generation_id,
        error,
    )

    if _is_current_generation(generation_id):
        _set_speaking(False)


def _complete_generation(generation_id: int) -> None:
    state = _get_state(generation_id)
    if state is None:
        return

    with _states_lock:
        if state.playback_finished:
            return

        state.playback_finished = True

    state.speech_complete_event.set()

    if _is_current_generation(generation_id):
        _set_speaking(False)
        LOGGER.debug("Playback finished (generation=%s).", generation_id)


# ============================================================================
# GENERATION WORKER
# ============================================================================

_shutdown_event = threading.Event()
_workers_started = False
_workers_lock = threading.RLock()

_generation_worker: threading.Thread | None = None
_playback_worker: threading.Thread | None = None


def _generation_worker_loop() -> None:
    while not _shutdown_event.is_set():
        try:
            request = _generation_queue.get(timeout=WORKER_POLL)
        except queue.Empty:
            continue

        try:
            generation_id = request.generation_id

            if not _is_current_generation(generation_id):
                continue

            state = _get_state(generation_id)
            if state is None or state.cancelled:
                continue

            model = _tts
            voice_state = _voice_state

            if model is None or voice_state is None:
                raise RuntimeError("Pocket TTS is not initialized.")

            # Pocket TTS returns a stream of audio arrays. Feed each array
            # directly into the playback queue as soon as it is available.
            audio_stream = model.generate_audio_stream(
                model_state=voice_state,
                text_to_generate=request.text,
                copy_state=True,
            )

            for audio in audio_stream:
                if _shutdown_event.is_set():
                    break

                if not _is_current_generation(generation_id):
                    break

                state = _get_state(generation_id)
                if state is None or state.cancelled:
                    break

                pcm = _audio_to_pcm(audio)
                if pcm.size == 0:
                    continue

                # Bounded queue with cancellation-aware backpressure.
                while not _shutdown_event.is_set():
                    if not _is_current_generation(generation_id):
                        break

                    state = _get_state(generation_id)
                    if state is None or state.cancelled:
                        break

                    try:
                        _audio_queue.put(
                            (generation_id, pcm),
                            timeout=QUEUE_WAIT,
                        )
                        break
                    except queue.Full:
                        continue

                else:
                    break

                state = _get_state(generation_id)
                if state is not None:
                    state.audio_chunks_enqueued += 1

                _audio_event.set()

            state = _get_state(generation_id)
            if (
                state is not None
                and _is_current_generation(generation_id)
                and not state.cancelled
            ):
                state.generated_chunks += 1

        except Exception as exc:
            if _is_current_generation(request.generation_id):
                _fail_generation(request.generation_id, exc)
        finally:
            _generation_queue.task_done()

            generation_id = request.generation_id
            state = _get_state(generation_id)

            if (
                state is not None
                and _is_current_generation(generation_id)
                and state.generated_chunks >= state.total_chunks
            ):
                state.generation_finished = True
                _audio_event.set()


# ============================================================================
# PLAYBACK WORKER
# ============================================================================

def _playback_worker_loop() -> None:
    active_generation: int | None = None
    process: subprocess.Popen[bytes] | None = None

    while not _shutdown_event.is_set():
        try:
            generation_id, pcm = _audio_queue.get(timeout=WORKER_POLL)
        except queue.Empty:
            current_id = _current_generation_id()
            state = _get_state(current_id)

            # Generation is finished and there is no buffered PCM left.
            # Because aplay remains open, there is no process shutdown here.
            if (
                state is not None
                and state.generation_finished
                and not state.generation_failed
                and not state.cancelled
                and _audio_queue.empty()
                and state.audio_chunks_enqueued > 0
            ):
                if active_generation == current_id:
                    _complete_generation(current_id)
                    active_generation = None

            continue

        try:
            if not _is_current_generation(generation_id):
                continue

            state = _get_state(generation_id)
            if (
                state is None
                or state.cancelled
                or state.generation_failed
            ):
                continue

            if active_generation != generation_id:
                # A new generation replaces any previous one. Normally the
                # old process has already been stopped by speak_async/stop.
                process = _get_audio_process()
                active_generation = generation_id
                state.playback_started = True

            if process is None or process.poll() is not None:
                process = _get_audio_process()

            stdin = process.stdin
            if stdin is None:
                raise RuntimeError("aplay stdin is unavailable.")

            stdin.write(pcm.tobytes())
            stdin.flush()

        except (BrokenPipeError, OSError, ValueError) as exc:
            LOGGER.error("Playback pipe failed: %s", exc)

            state = _get_state(generation_id)
            if state is not None and _is_current_generation(generation_id):
                _fail_generation(generation_id, exc)

            _stop_audio_process()
            process = None
            active_generation = None

        except Exception as exc:
            LOGGER.exception("Playback error.")

            state = _get_state(generation_id)
            if state is not None and _is_current_generation(generation_id):
                _fail_generation(generation_id, exc)

            _stop_audio_process()
            process = None
            active_generation = None

        finally:
            _audio_queue.task_done()
            _audio_event.set()

    # Clean worker exit.
    _stop_audio_process()


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

        LOGGER.debug("TTS workers started.")


# ============================================================================
# PUBLIC SPEAK API
# ============================================================================

_speak_lock = threading.RLock()


def speak_async(
    text: str,
    on_complete: Callable[[], None] | None = None,
) -> int | None:
    """
    Start speech asynchronously.

    New speech replaces any speech currently being generated or played.

    Returns:
        generation id, or None when there is no speakable text.
    """
    text = str(text or "").strip()
    if not text:
        return None

    chunks = _split_sentences(text)
    if not chunks:
        return None

    # Model initialization is intentionally outside _speak_lock so concurrent
    # callers do not serialize on the model load unnecessarily.
    preload()
    _ensure_workers()

    with _speak_lock:
        old_generation = _current_generation_id()
        generation_id = _next_generation_id()

        # Invalidate old work before touching queues/processes.
        if old_generation:
            _cancel_state(old_generation)

        _clear_generation_queue()
        _clear_audio_queue()

        # This is the important "barge in" operation: immediately stop
        # anything currently audible.
        _stop_audio_process()

        state = GenerationState(
            generation_id=generation_id,
            total_chunks=len(chunks),
        )
        _install_state(state)

        _set_speaking(True)

        for sequence, chunk in enumerate(chunks):
            request = SpeechRequest(
                generation_id=generation_id,
                sequence=sequence,
                text=chunk,
            )

            try:
                _generation_queue.put_nowait(request)
            except queue.Full:
                state.cancelled = True
                state.generation_failed = True
                state.generation_finished = True
                state.error = "TTS generation queue is full."
                state.speech_complete_event.set()
                _set_speaking(False)
                _clear_generation_queue()
                return None

        _audio_event.set()

    if on_complete is not None:
        def wait_for_completion() -> None:
            state.speech_complete_event.wait()

            if not _is_current_generation(generation_id):
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
                LOGGER.exception("Completion callback failed.")

        threading.Thread(
            target=wait_for_completion,
            name=f"ZOE-TTS-Completion-{generation_id}",
            daemon=True,
        ).start()

    return generation_id


def speak(
    text: str,
    on_complete: Callable[[], None] | None = None,
) -> bool:
    """
    Synchronous speech.

    Returns True only when the complete utterance was actually played.
    """
    generation_id = speak_async(
        text,
        on_complete=on_complete,
    )

    if generation_id is None:
        return False

    state = _get_state(generation_id)
    if state is None:
        return False

    state.speech_complete_event.wait()

    return (
        _is_current_generation(generation_id)
        and state.playback_finished
        and not state.cancelled
        and not state.generation_failed
    )


# ============================================================================
# STOP / SHUTDOWN
# ============================================================================

def stop() -> None:
    """Immediately cancel generation and stop currently audible speech."""
    with _speak_lock:
        current_generation = _current_generation_id()

        # Invalidate first so a generator cannot enqueue more audio after the
        # process is stopped.
        _next_generation_id()

        if current_generation:
            _cancel_state(current_generation)

        _set_speaking(False)

        _clear_generation_queue()
        _clear_audio_queue()
        _stop_audio_process()

        _audio_event.set()


def shutdown() -> None:
    """Stop speech, terminate workers, and release the audio process."""
    global _workers_started
    global _generation_worker
    global _playback_worker

    with _speak_lock:
        _set_speaking(False)

        current_generation = _current_generation_id()
        _next_generation_id()

        if current_generation:
            _cancel_state(current_generation)

        _clear_generation_queue()
        _clear_audio_queue()

        _shutdown_event.set()
        _audio_event.set()

        _stop_audio_process()

    current_thread = threading.current_thread()

    with _workers_lock:
        generation_worker = _generation_worker
        playback_worker = _playback_worker

    for worker in (generation_worker, playback_worker):
        if worker is not None and worker is not current_thread:
            worker.join(timeout=1.5)

    with _workers_lock:
        _generation_worker = None
        _playback_worker = None
        _workers_started = False

    LOGGER.info("Shutdown complete.")


# ============================================================================
# OPTIONAL DIAGNOSTICS
# ============================================================================

def get_status() -> dict[str, object]:
    """Return lightweight runtime diagnostics for the ZOE UI/backend."""
    generation_id = _current_generation_id()
    state = _get_state(generation_id)

    with _audio_process_lock:
        audio_process_alive = (
            _audio_process is not None
            and _audio_process.poll() is None
        )

    return {
        "speaking": is_speaking(),
        "generation_id": generation_id,
        "generation_queue": _generation_queue.qsize(),
        "audio_queue": _audio_queue.qsize(),
        "audio_process_alive": audio_process_alive,
        "model_loaded": _tts is not None and _voice_state is not None,
        "generation_finished": (
            state.generation_finished if state else False
        ),
        "generation_failed": (
            state.generation_failed if state else False
        ),
        "playback_started": (
            state.playback_started if state else False
        ),
        "playback_finished": (
            state.playback_finished if state else False
        ),
        "error": state.error if state else "",
    }


# ============================================================================
# TEST
# ============================================================================

def _test() -> None:
    preload()

    text = (
        "Hello Sir. This is the ZOE Pocket TTS production test. "
        "The CPU and GPU should be spoken naturally. "
        "The current temperature is 27 degrees Celsius. "
        "The server is available at https://example.com/api/v1. "
        "The email is zoe@example.com. "
        "You received a security code 4413796. "
        "The code should be spoken digit by digit. "
        "Playback should finish naturally."
    )

    LOGGER.info("Test starting...")
    started = time.monotonic()

    success = speak(text)

    LOGGER.info(
        "Test complete: success=%s elapsed=%.2fs",
        success,
        time.monotonic() - started,
    )


atexit.register(shutdown)


if __name__ == "__main__":
    try:
        _test()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted.")
    finally:
        shutdown()
