from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import atexit
import logging
import os
import queue
import re
import sys
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

# Unbounded on purpose: it only ever holds the TEXT chunks of the current
# utterance (cleared on every new speak/stop). A bounded queue silently
# dropped any reply longer than a dozen sentences. Audio backpressure is
# handled by _audio_queue, which stays small.
_generation_queue: queue.Queue[SpeechRequest] = queue.Queue()

# Audio is stored as FLOAT32 end to end (Pocket TTS -> float32 -> sounddevice),
# which avoids int16 round trips and extra copying.
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
#
# Pocket TTS reads numerals unreliably (digit by digit, mangled clock times,
# unit abbreviations as letters).  So numbers, times, dates, currency and
# units are converted to plain spoken English BEFORE they reach the model.

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = [
    (10**12, "trillion"), (10**9, "billion"),
    (10**6, "million"), (10**3, "thousand"),
]
_DIGITS = {str(i): w for i, w in enumerate(_ONES[:10])}
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]

# Positive number, optional thousands commas, optional decimals.
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def _spell_digits(s: str) -> str:
    return " ".join(_DIGITS[d] for d in s if d in _DIGITS)


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (" " + _ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + _int_to_words(r) if r else "")
    for value, name in _SCALES:
        if n >= value:
            q, r = divmod(n, value)
            return (_int_to_words(q) + " " + name
                    + (" " + _int_to_words(r) if r else ""))
    return str(n)


def _number_to_words(raw: str) -> str:
    """'1,234.5' -> 'one thousand two hundred thirty four point five'."""
    s = raw.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    grouped = "," in s
    s = s.replace(",", "")
    whole, _, frac = s.partition(".")
    whole = whole or "0"

    # Codes / phone numbers / leading zeros are read digit by digit.
    if not grouped and not frac and (
        len(whole) >= 7 or (len(whole) > 1 and whole[0] == "0")
    ):
        out = _spell_digits(whole)
    else:
        out = _int_to_words(int(whole))
        if frac:
            out += " point " + _spell_digits(frac)

    return ("minus " + out) if neg else out


def _pairs_to_words(digits: str) -> str:
    """Years and model numbers: 2026 -> twenty twenty six, 480 -> four eighty."""
    n = int(digits)

    if len(digits) == 4:
        if n % 1000 == 0:
            return _int_to_words(n)
        if 2000 <= n <= 2009:
            return "two thousand " + _ONES[n - 2000]
        a, b = divmod(n, 100)
        if b == 0:
            return _int_to_words(a) + " hundred"
        if b < 10:
            return _int_to_words(a) + " oh " + _ONES[b]
        return _int_to_words(a) + " " + _int_to_words(b)

    if len(digits) == 3:
        a, b = divmod(n, 100)
        if b == 0:
            return _int_to_words(n)
        if b < 10:
            return _ONES[a] + " oh " + _ONES[b]
        return _ONES[a] + " " + _int_to_words(b)

    return _number_to_words(digits)


_ORD_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def _ordinal_words(n: int) -> str:
    words = _int_to_words(n).split(" ")
    last = words[-1]
    if last in _ORD_IRREGULAR:
        words[-1] = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _clean_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)

    # "3 * 4" / "3*4" is multiplication, not emphasis. Handle it first.
    text = re.sub(r"(?<=\d)\s?\*\s?(?=\d)", " times ", text)

    # Strip emphasis asterisks only (attached to a word), never a lone " * ".
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\*+(?=\S)|(?<=\S)\*+", "", text)

    # Only strip emphasis underscores. snake_case / emails / URLs keep theirs
    # until they have been handled (final cleanup turns leftovers into spaces).
    text = re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)
    return text


# ---------------------------------------------------------------------------
# Emails / URLs / security codes
# ---------------------------------------------------------------------------

_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


def _replace_emails(text: str) -> str:
    return _EMAIL.sub(
        lambda m: (
            m.group(0)
            .replace("@", " at ")
            .replace(".", " dot ")
            .replace("_", " underscore ")
            .replace("-", " dash ")
        ),
        text,
    )


_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)


def _replace_urls(text: str) -> str:

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        trailing = ""
        while value and value[-1] in ".,!?;:)":
            trailing = value[-1] + trailing
            value = value[:-1]
        value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^www\.", "", value, flags=re.IGNORECASE)
        value = (
            value.replace("/", " slash ")
            .replace("_", " underscore ")
            .replace("-", " dash ")
            .replace(".", " dot ")
        )
        return value + trailing

    return _URL.sub(replace, text)


# Codes are ONLY read digit by digit when a code-like phrase comes right
# before them ("login code is 125012", "OTP: 4821", "PIN 1234").
# Group 2 is the connector ("is", ":", ...) and is KEPT in the output.
# (?!\d) instead of \b so a missing space ("125012The") still matches.
_SECURITY_CODE = re.compile(
    r"\b("
    r"(?:security|verification|verify|authentication|auth|access|login|"
    r"confirmation|reset|activation|sign[- ]?in)\s+code|"
    r"one[- ]time\s+(?:password|code|passcode)|otp|passcode|pin"
    r")(\s*(?:(?:is|was)\s*)?[:#-]?\s*)(\d{4,8})(?!\d)",
    re.IGNORECASE,
)


def _replace_security_codes(text: str) -> str:

    def replace(m: re.Match[str]) -> str:
        gap = m.group(2)
        if not gap or not gap[-1].isspace():
            gap += " "
        return m.group(1) + gap + _spell_digits(m.group(3))

    return _SECURITY_CODE.sub(replace, text)


# ---------------------------------------------------------------------------
# Special words / pronunciation overrides
# ---------------------------------------------------------------------------
#
# Keys MUST be UPPERCASE; matching is case-insensitive ("Hasin", "hasin" and
# "HASIN" all hit the "HASIN" key). The value is what Pocket TTS will be fed.
# Tweak the respelling by ear if a name still sounds off.

_ACRONYMS = {
    "ZOE": "Zoe",
    "2FA": "two factor authentication",
    "HASIN": "Ha-sin",
    "VS": "versus",
}

_ACRONYM_PATTERN = re.compile(
    r"\b(" + "|".join(
        re.escape(k) for k in sorted(_ACRONYMS, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)


def _replace_acronyms(text: str) -> str:
    return _ACRONYM_PATTERN.sub(
        lambda m: _ACRONYMS.get(m.group(1).upper(), m.group(1)), text
    )


# ---------------------------------------------------------------------------
# Acronyms spelled letter by letter
# ---------------------------------------------------------------------------
#
# Pocket TTS tries to read unknown letter groups as words ("API" -> "epy").
# These are spaced out so each letter is pronounced:
#   API -> "A P I",  LED -> "L E D",  APIs -> "A P Is"
#
# To add one, just add it to the right set below.

# Safe in any case: none of these are ordinary English words.
# (write them lowercase)
_SPELLED_ANY_CASE = {
    # computing / hardware
    "api", "cpu", "gpu", "npu", "tpu", "usb", "hdmi", "ssd", "hdd", "lcd",
    "oled", "gpio", "nvme", "sata", "wlan", "pcb", "adc", "dac", "pwm",
    "rgb", "nfc", "gps", "dpi", "ppi", "sms", "mms",
    # software / web
    "html", "css", "xml", "csv", "php", "js", "ssh", "ssl", "tls", "tcp",
    "udp", "url", "uri", "http", "https", "dns", "vpn", "cdn", "mqtt",
    "pdf", "sdk", "cli", "gui", "ui", "ux", "ide", "jwt", "uuid", "orm",
    "aws", "esp", "ssid",
    # AI / speech
    "tts", "stt", "asr", "llm", "ocr",
    # graphics cards
    "rtx", "gtx",
}

# Only when written in capitals ("OS", "ID"), because "os", "id", "led",
# "ip", "us", "hr" and so on are also ordinary words or fragments.
# (write them UPPERCASE)  "RAM" is intentionally NOT here: people say "ram".
_SPELLED_UPPER_ONLY = {
    "AI", "OS", "ID", "PC", "TV", "LED", "IP", "ML", "UK", "US", "EU",
    "UN", "HD", "UV", "AC", "DC", "EV", "VR", "AR", "QA", "PR", "HR",
    "DIY", "FYI", "CEO", "CTO", "CFO", "ATM", "CD", "DVD",
}

# Said as a word, or needs a custom respelling. Delete a line to turn it off.
_SPELLED_OVERRIDES = {
    "wifi": "Wi Fi",
    "json": "Jason",
    "sql": "sequel",
    "oauth": "oh auth",
    "cors": "cores",
    "yaml": "yammel",
}

_SPELLED_ANY_RE = re.compile(
    r"\b("
    + "|".join(
        re.escape(k)
        for k in sorted(
            _SPELLED_ANY_CASE | set(_SPELLED_OVERRIDES), key=len, reverse=True
        )
    )
    + r")(s?)\b",
    re.IGNORECASE,
)

_SPELLED_UPPER_RE = re.compile(
    r"\b("
    + "|".join(
        re.escape(k) for k in sorted(_SPELLED_UPPER_ONLY, key=len, reverse=True)
    )
    + r")(s?)\b"
)


def _spell_acronym(word: str, plural: str) -> str:
    override = _SPELLED_OVERRIDES.get(word.lower())
    if override:
        return override + plural

    letters = list(word.upper())

    if plural:
        # "A P Is" gets read as the word "is". Respell the plural ending:
        #   APIs -> "A P Eyes",  GPUs -> "G P U's",  CEOs -> "C E O's"
        # Tune by ear: change "Eyes" or switch "'s" to "z" if needed.
        if letters[-1] == "I":
            letters[-1] = "Eyes"
        else:
            letters[-1] = letters[-1] + "'s"

    return " ".join(letters)


def _replace_spelled_acronyms(text: str) -> str:
    text = _SPELLED_ANY_RE.sub(
        lambda m: _spell_acronym(m.group(1), m.group(2)), text
    )
    text = _SPELLED_UPPER_RE.sub(
        lambda m: _spell_acronym(m.group(1), m.group(2)), text
    )
    return text


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_ISO_DATE = re.compile(
    r"\b(\d{4})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b"
)


def _replace_iso_dates(text: str) -> str:
    return _ISO_DATE.sub(
        lambda m: (
            f"{_MONTH_NAMES[int(m.group(2)) - 1]} "
            f"{_ordinal_words(int(m.group(3)))}, "
            f"{_pairs_to_words(m.group(1))}"
        ),
        text,
    )


_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|"
    r"Oct|Nov|Dec)"
)
_YEAR = r"(?:1[5-9]|20)\d\d"

# "October 4, 2026" / "Oct 2026"
_YEAR_AFTER_MONTH = re.compile(
    r"\b(" + _MONTHS + r"\.?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?)(" + _YEAR + r")\b"
)
# "in 2026", "since 1999"
_YEAR_CONTEXT = re.compile(
    r"\b(in|since|until|till|during|year)\s+(" + _YEAR + r")\b",
    re.IGNORECASE,
)


def _replace_years(text: str) -> str:
    text = _YEAR_AFTER_MONTH.sub(
        lambda m: m.group(1) + _pairs_to_words(m.group(2)), text
    )
    text = _YEAR_CONTEXT.sub(
        lambda m: m.group(1) + " " + _pairs_to_words(m.group(2)), text
    )
    return text


# ---------------------------------------------------------------------------
# Times
# ---------------------------------------------------------------------------

def _clock_words(hour: int, minute: int) -> str:
    out = _int_to_words(hour)
    if minute:
        out += (" oh " + _ONES[minute]) if minute < 10 else (
            " " + _int_to_words(minute)
        )
    return out


# 10 PM / 10:30 PM / 10:30 p.m.
_TIME_12 = re.compile(
    r"(?<![\d:.])(\d{1,2})(?::([0-5]\d))?\s*(AM|PM|A\.M\.|P\.M\.)(?!\w)",
    re.IGNORECASE,
)


def _replace_times_12(text: str) -> str:

    def replace(m: re.Match[str]) -> str:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        if not 1 <= hour <= 12:
            return m.group(0)
        period = "A M" if m.group(3).upper().startswith("A") else "P M"
        return f"{_clock_words(hour, minute)} {period}"

    return _TIME_12.sub(replace, text)


# 22:35 / 07:05 / 1:30
_TIME_24 = re.compile(r"(?<![\d:.])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")


def _replace_times_24(text: str) -> str:

    def replace(m: re.Match[str]) -> str:
        hs = m.group(1)
        hour = int(hs)
        minute = int(m.group(2))

        if hour == 0 and minute == 0:
            return "midnight"
        if hour == 12 and minute == 0:
            return "noon"

        if hour > 12:
            hour12, period = hour - 12, "P M"
        elif hour == 0:
            hour12, period = 12, "A M"
        elif hour == 12:
            hour12, period = 12, "P M"
        else:
            hour12 = hour
            # "07:30" is obviously morning; "1:30" is ambiguous -> no suffix.
            period = "A M" if hs.startswith("0") else ""

        out = _clock_words(hour12, minute)
        if minute == 0 and not period:
            out += " o'clock"
        return f"{out} {period}".strip()

    return _TIME_24.sub(replace, text)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

# symbol -> (major singular, major plural, minor singular, minor plural)
_CURRENCY_NAMES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "৳": ("taka", "taka", "poisha", "poisha"),
}

_CURRENCY = re.compile(
    r"(?:([$€£৳])|(?<!\w)(?:Tk|BDT)\.?)\s?(" + _NUM + r")"
    r"(?:\s?(thousand|million|billion|trillion)\b)?",
    re.IGNORECASE,
)


def _replace_currency(text: str) -> str:

    def replace(m: re.Match[str]) -> str:
        sym = m.group(1) or "৳"
        raw = m.group(2)
        scale = m.group(3)
        maj_s, maj_p, min_s, min_p = _CURRENCY_NAMES[sym]

        if scale:
            return f"{_number_to_words(raw)} {scale.lower()} {maj_p}"

        whole, _, frac = raw.replace(",", "").partition(".")
        wn = int(whole)
        cents = int((frac + "00")[:2]) if frac else 0

        parts = []
        if wn or not cents:
            parts.append(f"{_int_to_words(wn)} {maj_s if wn == 1 else maj_p}")
        if cents:
            parts.append(f"{_int_to_words(cents)} {min_s if cents == 1 else min_p}")
        return " and ".join(parts)

    return _CURRENCY.sub(replace, text)


# ---------------------------------------------------------------------------
# Percent, ranges, temperature
# ---------------------------------------------------------------------------

_PERCENT = re.compile(r"(?<![\w.])(-?" + _NUM + r")\s?%")


def _replace_percentages(text: str) -> str:
    return _PERCENT.sub(
        lambda m: f"{_number_to_words(m.group(1))} percent", text
    )


# 10-20 / 10 – 20   (a plain " - " is left alone: could be subtraction)
# Note: this is lossy for things like "1-800"; it is kept because ranges
# ("10-20 minutes") are far more common in assistant replies.
_RANGE = re.compile(
    r"(?<![\w.-])(\d{1,4}(?:\.\d+)?)(?:-|\s?[–—]\s?)(\d{1,4}(?:\.\d+)?)(?![\d-])"
)


def _replace_ranges(text: str) -> str:
    return _RANGE.sub(lambda m: f"{m.group(1)} to {m.group(2)}", text)


_TEMPERATURE = re.compile(
    r"(?<![\w.])(-?" + _NUM + r")\s?(?:°\s?([CcFf])|℃|℉)(?!\w)"
)


def _replace_temperatures(text: str) -> str:

    def replace(m: re.Match[str]) -> str:
        whole = m.group(0)
        scale = (m.group(2) or ("C" if "℃" in whole else "F")).upper()
        name = "Celsius" if scale == "C" else "Fahrenheit"
        return f"{_number_to_words(m.group(1))} degrees {name}"

    text = _TEMPERATURE.sub(replace, text)
    return text.replace("°", " degrees ")


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# Case-insensitive, optional space: "27 kph", "27KPH", "27 km/h"
_UNITS_LOOSE = {
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "kmph": ("kilometer per hour", "kilometers per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "mph": ("mile per hour", "miles per hour"),
    "m/s": ("meter per second", "meters per second"),
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "kg": ("kilogram", "kilograms"),
    "mg": ("milligram", "milligrams"),
    "lbs": ("pound", "pounds"),
    "lb": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "ml": ("milliliter", "milliliters"),
    "kb": ("kilobyte", "kilobytes"),
    "mb": ("megabyte", "megabytes"),
    "gb": ("gigabyte", "gigabytes"),
    "tb": ("terabyte", "terabytes"),
    "kbps": ("kilobit per second", "kilobits per second"),
    "mbps": ("megabit per second", "megabits per second"),
    "gbps": ("gigabit per second", "gigabits per second"),
    "hz": ("hertz", "hertz"),
    "khz": ("kilohertz", "kilohertz"),
    "mhz": ("megahertz", "megahertz"),
    "ghz": ("gigahertz", "gigahertz"),
    "kw": ("kilowatt", "kilowatts"),
    "kwh": ("kilowatt hour", "kilowatt hours"),
    "mah": ("milliamp hour", "milliamp hours"),
    "kcal": ("calorie", "calories"),
    "ms": ("millisecond", "milliseconds"),
    "sec": ("second", "seconds"),
    "secs": ("second", "seconds"),
    "min": ("minute", "minutes"),
    "mins": ("minute", "minutes"),
    "hr": ("hour", "hours"),
    "hrs": ("hour", "hours"),
    "fps": ("frame per second", "frames per second"),
    "rpm": ("revolution per minute", "revolutions per minute"),
    "db": ("decibel", "decibels"),
    "hpa": ("hectopascal", "hectopascals"),
    "kpa": ("kilopascal", "kilopascals"),
    "psi": ("P S I", "P S I"),
    "ppm": ("part per million", "parts per million"),
    "lux": ("lux", "lux"),
}

# Case-sensitive, NO space: "5m", "10s", "12V", "3A"
_UNITS_TIGHT = {
    "m": ("meter", "meters"),
    "g": ("gram", "grams"),
    "s": ("second", "seconds"),
    "h": ("hour", "hours"),
    "V": ("volt", "volts"),
    "W": ("watt", "watts"),
    "A": ("amp", "amps"),
}


def _unit_regex(table: dict, spaced: bool, flags: int) -> re.Pattern[str]:
    alt = "|".join(re.escape(k) for k in sorted(table, key=len, reverse=True))
    gap = r"\s?" if spaced else ""
    return re.compile(
        r"(?<![\w.])(-?" + _NUM + r")" + gap + r"(" + alt + r")(?!\w)", flags
    )


_UNIT_LOOSE_RE = _unit_regex(_UNITS_LOOSE, True, re.IGNORECASE)
_UNIT_TIGHT_RE = _unit_regex(_UNITS_TIGHT, False, 0)


def _unit_replacer(table: dict, lower: bool):

    def replace(m: re.Match[str]) -> str:
        num = m.group(1)
        unit = m.group(2).lower() if lower else m.group(2)
        singular, plural = table[unit]
        word = singular if num == "1" else plural
        return f"{_number_to_words(num)} {word}"

    return replace


def _replace_units(text: str) -> str:
    text = _UNIT_LOOSE_RE.sub(_unit_replacer(_UNITS_LOOSE, True), text)
    text = _UNIT_TIGHT_RE.sub(_unit_replacer(_UNITS_TIGHT, False), text)
    return text


# A unit with no number in front of it ("what is the speed in kph?")
_BARE_UNIT = re.compile(r"(?<![\w/])(km/h|kmph|kph|mph)(?![\w/])", re.IGNORECASE)


def _replace_bare_units(text: str) -> str:
    return _BARE_UNIT.sub(
        lambda m: _UNITS_LOOSE[m.group(1).lower()][1], text
    )


# ---------------------------------------------------------------------------
# Ordinals, versions, IPs, model names
# ---------------------------------------------------------------------------

_ORDINAL = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})*)(st|nd|rd|th)\b", re.IGNORECASE)


def _replace_ordinals(text: str) -> str:
    return _ORDINAL.sub(
        lambda m: _ordinal_words(int(m.group(1).replace(",", ""))), text
    )


# v2.5.1 -> version two point five point one
_VERSION = re.compile(r"\b[vV](\d+(?:\.\d+)*)\b")


def _replace_versions(text: str) -> str:
    return _VERSION.sub(
        lambda m: "version " + " point ".join(
            _int_to_words(int(p)) for p in m.group(1).split(".")
        ),
        text,
    )


# 192.168.0.101 -> digits with "dot";  1.2.3 -> one point two point three
_DOTTED = re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?![\w])")


def _replace_dotted_numbers(text: str) -> str:

    def replace(m: re.Match[str]) -> str:
        parts = m.group(0).split(".")
        if len(parts) == 4 and all(len(p) <= 3 and int(p) <= 255 for p in parts):
            return " dot ".join(_spell_digits(p) for p in parts)
        return " point ".join(_number_to_words(p) for p in parts)

    return _DOTTED.sub(replace, text)


# Hardware / model names only: an UPPERCASE prefix (T480, RTX4090, USB3, MP3)
# or a known product word (iPhone15, Pixel8). Case-sensitive on purpose, so
# ordinary words and usernames like "win11" or "user42" are left alone.
_MODEL_NAME = re.compile(
    r"\b([A-Z]{1,5}|iPhone|iPad|iMac|Pixel|Galaxy)(\d+)\b"
)


def _replace_model_names(text: str) -> str:

    def replace(m: re.Match[str]) -> str:
        digits = m.group(2)
        if len(digits) in (3, 4) and digits[0] != "0":
            spoken = _pairs_to_words(digits)
        else:
            spoken = _number_to_words(digits)
        return f"{m.group(1)} {spoken}"

    return _MODEL_NAME.sub(replace, text)


# 1080p -> ten eighty p
_RESOLUTION = re.compile(r"(?<![\w.])(\d{3,4})([pi])\b")


def _replace_resolutions(text: str) -> str:
    return _RESOLUTION.sub(
        lambda m: f"{_pairs_to_words(m.group(1))} {m.group(2)}", text
    )


# 5G -> five G | 4K -> four K | 3D -> three D
_DIGIT_CAPS = re.compile(r"(?<![\w.])(\d+)([A-Z]{1,3})(?!\w)")


def _replace_digit_caps(text: str) -> str:
    return _DIGIT_CAPS.sub(
        lambda m: f"{_number_to_words(m.group(1))} {' '.join(m.group(2))}",
        text,
    )


# ---------------------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------------------

_GENERIC_NUMBER = re.compile(r"(?<![\w.])(-?" + _NUM + r")(?!\w)")


def _replace_remaining_numbers(text: str) -> str:
    return _GENERIC_NUMBER.sub(lambda m: _number_to_words(m.group(1)), text)


def _replace_symbols(text: str) -> str:
    text = re.sub(r"#\s?(?=\d)", "number ", text)
    text = re.sub(r"~\s?(?=\d)", "about ", text)
    text = text.replace("×", " times ")
    text = text.replace("&", " and ")
    return text


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
    text = text.replace("−", "-")

    # Structured things first. Security codes run before any generic number
    # handling, so only code-context numbers are read digit by digit.
    text = _replace_emails(text)
    text = _replace_urls(text)
    text = _replace_security_codes(text)
    text = _replace_acronyms(text)

    # Dates and times (before anything touches ':' or '-').
    text = _replace_iso_dates(text)
    text = _replace_times_12(text)
    text = _replace_times_24(text)
    text = _replace_years(text)

    # Quantities.
    text = _replace_currency(text)
    text = _replace_percentages(text)
    text = _replace_ranges(text)
    text = _replace_temperatures(text)
    text = _replace_units(text)
    text = _replace_bare_units(text)

    # Names and identifiers.
    text = _replace_ordinals(text)
    text = _replace_versions(text)
    text = _replace_dotted_numbers(text)
    text = _replace_resolutions(text)
    text = _replace_model_names(text)
    text = _replace_digit_caps(text)

    # Whatever numbers are left.
    text = _replace_symbols(text)
    text = _replace_remaining_numbers(text)

    # Last, so "USB3" -> "USB three" -> "U S B three".
    text = text.replace("_", " ")
    text = _replace_spelled_acronyms(text)

    text = re.sub(r"\s+", " ", text).strip()

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
# CLEANUP
# ============================================================================

atexit.register(shutdown)


# ============================================================================
# QUICK CHECK
# ============================================================================
#
#   python tts.py            speaks a sample line (real model + speakers)
#   python tts.py --text     prints what the normalizer produces (no audio)

if __name__ == "__main__":

    SAMPLE = (
        "Hello Hasin. The time is 22:35, and the wind is 27 kph. "
        "Your Meta login code is 125012. "
        "The API call finished. I have 16 GB of RAM and 3 APIs."
    )

    try:
        if "--text" in sys.argv:
            print(_normalize_speech(SAMPLE))
        else:
            speak(SAMPLE)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()