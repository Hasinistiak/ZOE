from __future__ import annotations

import atexit
import logging
import os
import queue
import re
import tempfile
import threading
import time
import wave
from contextlib import closing
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
from dotenv import load_dotenv

try:
    import sounddevice as sd
except OSError as exc:
    raise SystemExit(
        "[ZOE TTS] PortAudio library not found.\n"
        "Install: sudo pacman -S portaudio\n"
        f"Original error: {exc}"
    )

from cartesia import Cartesia

load_dotenv()

# ============================================================================
# CONFIG
# ============================================================================


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _parse_device(raw: str) -> int | str | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.lstrip("-").isdigit():
        return int(raw)
    return raw


def _parse_latency(raw: str) -> float | str:
    raw = raw.strip().lower() or "high"
    if raw in ("low", "high"):
        return raw
    try:
        return float(raw)
    except ValueError:
        return "high"


MODEL = os.getenv("ZOE_TTS_MODEL", "sonic-latest")
VOICE = os.getenv("ZOE_TTS_VOICE", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b")
LANGUAGE = "en"

SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes, pcm_s16le

MAX_CHUNK_LENGTH = _env_int("ZOE_TTS_CHUNK", 320, minimum=120)
MAX_AUDIO_QUEUE = _env_int("ZOE_TTS_QUEUE", 6, minimum=2)
AUDIO_LATENCY = _parse_latency(os.getenv("ZOE_TTS_LATENCY", "high"))
AUDIO_BLOCKSIZE = _env_int("ZOE_TTS_BLOCKSIZE", 0, minimum=0)
TTS_DEVICE = _parse_device(os.getenv("ZOE_TTS_DEVICE", ""))

REQUEST_TIMEOUT = _env_float("ZOE_TTS_TIMEOUT", 30.0, minimum=5.0)
MAX_ATTEMPTS = _env_int("ZOE_TTS_ATTEMPTS", 2, minimum=1)
RETRY_DELAY = 0.4

WORKER_POLL = 0.05
QUEUE_WAIT = 0.01
PLAY_SLICE = 2048  # frames per write (~46 ms); bounds interrupt latency

# ============================================================================
# LOGGING
# ============================================================================

LOGGER = logging.getLogger("zoe.tts")

if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[ZOE TTS] %(levelname)s: %(message)s"))
    LOGGER.addHandler(_handler)

_level = getattr(logging, os.getenv("ZOE_TTS_LOG_LEVEL", "INFO").strip().upper(), logging.INFO)
LOGGER.setLevel(_level if isinstance(_level, int) else logging.INFO)
LOGGER.propagate = False

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ============================================================================
# TEXT
# ============================================================================

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HAS_SPEECH = re.compile(r"\w", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _prepare_text(text: object) -> str:
    """Strip control characters and normalize whitespace."""
    try:
        text = str(text or "")
    except Exception:
        return ""

    text = _CONTROL_CHARS.sub(" ", text)
    return " ".join(text.split())


def _split_long_text(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for word in text.split():
        # A single "word" longer than the limit (URLs etc.) gets hard-split.
        while len(word) > MAX_CHUNK_LENGTH:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            chunks.append(word[:MAX_CHUNK_LENGTH])
            word = word[MAX_CHUNK_LENGTH:]

        if not word:
            continue

        additional = len(word) + (1 if current else 0)

        if current and length + additional > MAX_CHUNK_LENGTH:
            chunks.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += additional

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_sentences(text: object) -> list[str]:
    text = _prepare_text(text)
    if not text:
        return []

    chunks: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= MAX_CHUNK_LENGTH:
            chunks.append(sentence)
        else:
            chunks.extend(_split_long_text(sentence))

    # Drop chunks with nothing pronounceable (e.g. "...", "--"); they make
    # the API error out.
    return [c for c in chunks if _HAS_SPEECH.search(c)]


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

    speech_complete_event: threading.Event = field(default_factory=threading.Event)


# ============================================================================
# CARTESIA CLIENT
# ============================================================================

_client: Cartesia | None = None
_client_lock = threading.Lock()


def _get_client() -> Cartesia:
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        api_key = os.getenv("CARTESIA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY environment variable is missing.")

        LOGGER.info("Initializing Cartesia client...")

        try:
            client = Cartesia(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=1)
        except TypeError:
            # Older SDK without timeout/max_retries kwargs.
            client = Cartesia(api_key=api_key)

        _client = client
        LOGGER.info("Cartesia ready: model=%s voice=%s", MODEL, VOICE)
        return _client


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


def _get_state(generation_id: int) -> GenerationState | None:
    with _states_lock:
        return _states.get(generation_id)


def _install_state(state: GenerationState) -> None:
    with _states_lock:
        _states.clear()
        _states[state.generation_id] = state


def _cancel_state(generation_id: int) -> None:
    state = _get_state(generation_id)
    if state is None:
        return
    with _states_lock:
        state.cancelled = True
    state.speech_complete_event.set()


def _still_active(generation_id: int) -> bool:
    """True while this generation should keep producing / playing audio."""
    if _shutdown_event.is_set() or not _is_current(generation_id):
        return False
    state = _get_state(generation_id)
    return state is not None and not state.cancelled and not state.generation_failed


# ============================================================================
# QUEUES
# ============================================================================

_generation_queue: queue.Queue[SpeechRequest] = queue.Queue()

# Items are (generation_id, pcm_bytes) or (generation_id, None) as the
# end-of-generation sentinel.
_audio_queue: queue.Queue[tuple[int, bytes | None]] = queue.Queue(maxsize=MAX_AUDIO_QUEUE)


def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
        else:
            q.task_done()


def _clear_generation_queue() -> None:
    _drain(_generation_queue)


def _clear_audio_queue() -> None:
    _drain(_audio_queue)


def _put_audio(generation_id: int, item: bytes | None) -> bool:
    """Backpressured put. Returns False if the generation was abandoned."""
    while _still_active(generation_id):
        try:
            _audio_queue.put((generation_id, item), timeout=QUEUE_WAIT)
        except queue.Full:
            continue

        if item is not None:
            state = _get_state(generation_id)
            if state is not None:
                with _states_lock:
                    state.audio_chunks += 1
        return True
    return False


# ============================================================================
# AUDIO OUTPUT
# ============================================================================

# IMPORTANT: every PortAudio stream call (open/start/write/abort/close) happens
# on the playback thread ONLY. Aborting/closing a stream from another thread
# while write() is running segfaults PortAudio. Other threads interrupt speech
# by invalidating the generation; the playback thread notices within one slice
# and closes the stream itself.
_audio_stream: sd.OutputStream | None = None  # playback thread only
_stream_open = False  # status flag, safe to read from anywhere
_dirty_generation = 0  # playback thread only: gen whose audio may be buffered


def _close_quietly(obj, *method_names: str) -> None:
    if obj is None:
        return
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _open_stream(device: int | str | None) -> sd.OutputStream:
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=device,
        blocksize=AUDIO_BLOCKSIZE,
        latency=AUDIO_LATENCY,
    )
    try:
        stream.start()
    except Exception:
        _close_quietly(stream, "close")
        raise
    return stream


def _discard_stream() -> None:
    """Abort and close the stream. Playback thread only."""
    global _audio_stream, _stream_open

    stream = _audio_stream
    _audio_stream = None
    _stream_open = False
    _close_quietly(stream, "abort", "close")


def _get_audio_stream() -> sd.OutputStream:
    """Return an active stream, opening one if needed. Playback thread only."""
    global _audio_stream, _stream_open

    stream = _audio_stream
    if stream is not None:
        try:
            if stream.active:
                return stream
        except Exception:
            pass
        _discard_stream()

    LOGGER.debug(
        "Opening audio device=%s rate=%s",
        TTS_DEVICE if TTS_DEVICE is not None else "default",
        SAMPLE_RATE,
    )

    try:
        stream = _open_stream(TTS_DEVICE)
    except Exception as exc:
        if TTS_DEVICE is None:
            raise
        LOGGER.warning(
            "Audio device %r failed (%s); falling back to default device.",
            TTS_DEVICE,
            exc,
        )
        stream = _open_stream(None)

    _audio_stream = stream
    _stream_open = True
    return stream


def _discard_if_stale() -> None:
    """Close the stream if the audio in it belongs to an interrupted speech."""
    global _dirty_generation

    if _dirty_generation and not _still_active(_dirty_generation):
        _dirty_generation = 0
        _discard_stream()


def _play_audio(stream: sd.OutputStream, generation_id: int, audio: bytes) -> None:
    """Play PCM in small slices, bailing out immediately if interrupted."""
    usable = len(audio) - (len(audio) % SAMPLE_WIDTH)
    if usable <= 0:
        return

    samples = np.frombuffer(audio[:usable], dtype=np.int16)

    for start in range(0, samples.size, PLAY_SLICE):
        if not _still_active(generation_id):
            return
        piece = np.array(samples[start:start + PLAY_SLICE], dtype=np.int16)
        stream.write(piece.reshape(-1, 1))


def _flush_audio(stream: sd.OutputStream, generation_id: int) -> None:
    """Push silence equal to the output latency so the tail isn't cut off."""
    try:
        latency = float(stream.latency)
    except Exception:
        latency = 0.3

    frames = int((latency + 0.05) * SAMPLE_RATE)

    while frames > 0 and _still_active(generation_id):
        n = min(frames, PLAY_SLICE)
        stream.write(np.zeros((n, 1), dtype=np.int16))
        frames -= n


def list_audio_devices() -> str:
    try:
        lines = []
        for index, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                lines.append(
                    f"{index:>3}  {dev['name']}  "
                    f"(out={dev['max_output_channels']}, "
                    f"rate={int(dev['default_samplerate'])})"
                )
        return "\n".join(lines) or "No output devices found."
    except Exception as exc:
        return f"Could not query audio devices: {exc}"


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
# CARTESIA STREAMING
# ============================================================================


def _generate_audio_stream(text: str):
    return _get_client().tts.generate_sse(
        model_id=MODEL,
        transcript=text,
        voice=VOICE,
        language=LANGUAGE,
        output_format={
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
        },
    )


def _event_audio(event) -> bytes:
    audio = getattr(event, "audio", None)
    if audio is None:
        return b""
    if isinstance(audio, bytes):
        return audio
    try:
        return bytes(audio)
    except Exception:
        return b""


def _iter_pcm(text: str) -> Iterator[bytes]:
    """
    Yield sample-aligned PCM for one text chunk. Network chunks can split in
    the middle of a 16-bit sample, so leftover bytes carry into the next one.
    Always use via contextlib.closing() so the HTTP stream gets closed.
    """
    stream = None
    leftover = b""

    try:
        stream = _generate_audio_stream(text)

        for event in stream:
            etype = str(getattr(event, "type", "")).lower().rsplit(".", 1)[-1]

            if etype == "error":
                raise RuntimeError(
                    f"{getattr(event, 'title', 'error')}: {getattr(event, 'message', '')}"
                )
            if etype == "done":
                break
            if etype != "chunk":
                continue

            audio = _event_audio(event)
            if not audio:
                continue

            audio = leftover + audio
            usable = len(audio) - (len(audio) % SAMPLE_WIDTH)
            leftover = audio[usable:]

            if usable:
                yield audio[:usable]
    finally:
        _close_quietly(stream, "close")


# ============================================================================
# COMPLETION
# ============================================================================


def _fail_generation(generation_id: int, error: Exception | str) -> None:
    state = _get_state(generation_id)
    if state is None:
        return

    with _states_lock:
        if state.generation_failed:
            return
        state.generation_failed = True
        state.generation_finished = True
        state.error = str(error)

    state.speech_complete_event.set()

    if _is_current(generation_id):
        _set_speaking(False)

    LOGGER.error("Generation failed (%s): %s", generation_id, error)


def _complete_generation(generation_id: int) -> None:
    state = _get_state(generation_id)
    if state is None:
        return

    with _states_lock:
        if state.playback_finished:
            return
        state.playback_finished = True

    state.speech_complete_event.set()

    if _is_current(generation_id):
        _set_speaking(False)

    LOGGER.debug("Speech completed: generation=%s", generation_id)


# ============================================================================
# WORKERS
# ============================================================================

_shutdown_event = threading.Event()
_workers_lock = threading.RLock()

_generation_worker: threading.Thread | None = None
_playback_worker: threading.Thread | None = None


def _stream_chunk(generation_id: int, text: str, progress: list[int]) -> bool:
    """Stream one chunk into the audio queue. False if abandoned."""
    with closing(_iter_pcm(text)) as pcm_iter:
        for pcm in pcm_iter:
            if not _put_audio(generation_id, pcm):
                return False
            progress[0] += 1
    return True


def _process_request(request: SpeechRequest) -> None:
    gid = request.generation_id

    if not _still_active(gid):
        return

    attempt = 0
    while True:
        attempt += 1
        progress = [0]

        try:
            finished = _stream_chunk(gid, request.text, progress)
            break
        except Exception as exc:
            # Safe to retry only if nothing from this chunk was played/queued.
            if progress[0] == 0 and attempt < MAX_ATTEMPTS and _still_active(gid):
                LOGGER.warning("Chunk failed (%s); retrying...", exc)
                _shutdown_event.wait(RETRY_DELAY)
                continue
            _fail_generation(gid, exc)
            return

    if not finished:
        return

    state = _get_state(gid)
    if state is None:
        return

    with _states_lock:
        state.completed_chunks += 1
        last = state.completed_chunks >= state.total_chunks
        if last:
            state.generation_finished = True

    if last:
        # End-of-speech marker; playback worker finishes the generation.
        _put_audio(gid, None)


def _generation_worker_loop() -> None:
    while not _shutdown_event.is_set():
        try:
            request = _generation_queue.get(timeout=WORKER_POLL)
        except queue.Empty:
            continue
        except Exception:
            LOGGER.exception("Generation queue error.")
            time.sleep(0.1)
            continue

        try:
            _process_request(request)
        except Exception as exc:
            LOGGER.exception("Unexpected generation error.")
            try:
                _fail_generation(request.generation_id, exc)
            except Exception:
                pass
        finally:
            try:
                _generation_queue.task_done()
            except ValueError:
                pass


def _handle_audio(generation_id: int, audio: bytes | None) -> None:
    global _dirty_generation

    if not _still_active(generation_id):
        return

    state = _get_state(generation_id)
    if state is None:
        return

    try:
        if audio is None:
            # End of speech: let buffered audio finish, then close the stream.
            if state.playback_started and _audio_stream is not None:
                _flush_audio(_audio_stream, generation_id)
            if _still_active(generation_id):
                _dirty_generation = 0
                _discard_stream()
                _complete_generation(generation_id)
            return

        stream = _get_audio_stream()
        _dirty_generation = generation_id
        with _states_lock:
            state.playback_started = True
        _play_audio(stream, generation_id, audio)

    except Exception as exc:
        if _still_active(generation_id):
            _fail_generation(generation_id, exc)
        _dirty_generation = 0
        _discard_stream()


def _playback_worker_loop() -> None:
    global _dirty_generation

    while not _shutdown_event.is_set():
        _discard_if_stale()

        try:
            generation_id, audio = _audio_queue.get(timeout=WORKER_POLL)
        except queue.Empty:
            continue
        except Exception:
            LOGGER.exception("Audio queue error.")
            time.sleep(0.1)
            continue

        try:
            _handle_audio(generation_id, audio)
        except Exception:
            LOGGER.exception("Unexpected playback error.")
            if _is_current(generation_id):
                _fail_generation(generation_id, "unexpected playback error")
            _dirty_generation = 0
            _discard_stream()
        finally:
            try:
                _audio_queue.task_done()
            except ValueError:
                pass

        _discard_if_stale()

    _discard_stream()


def _ensure_workers() -> None:
    global _generation_worker
    global _playback_worker

    with _workers_lock:
        _shutdown_event.clear()

        if _generation_worker is None or not _generation_worker.is_alive():
            _generation_worker = threading.Thread(
                target=_generation_worker_loop,
                name="ZOE-Cartesia-Generator",
                daemon=True,
            )
            _generation_worker.start()

        if _playback_worker is None or not _playback_worker.is_alive():
            _playback_worker = threading.Thread(
                target=_playback_worker_loop,
                name="ZOE-Cartesia-Playback",
                daemon=True,
            )
            _playback_worker.start()


# ============================================================================
# SPEAK
# ============================================================================

_speak_lock = threading.RLock()


def _start_speech(
    text: str,
    on_complete: Callable[[], None] | None,
) -> GenerationState | None:
    try:
        chunks = _split_sentences(text)
        if not chunks:
            return None

        _get_client()  # validates API key early
        _ensure_workers()

        with _speak_lock:
            old_generation = _current_generation_id()
            generation_id = _next_generation_id()

            if old_generation:
                _cancel_state(old_generation)

            _clear_generation_queue()
            _clear_audio_queue()

            state = GenerationState(
                generation_id=generation_id,
                total_chunks=len(chunks),
            )
            _install_state(state)
            _set_speaking(True)

            for sequence, chunk in enumerate(chunks):
                _generation_queue.put_nowait(
                    SpeechRequest(generation_id, sequence, chunk)
                )
    except Exception:
        LOGGER.exception("Could not start speech.")
        _set_speaking(False)
        return None

    if on_complete is not None:

        def completion_waiter() -> None:
            state.speech_complete_event.wait()

            if not _is_current(state.generation_id):
                return
            if state.cancelled or state.generation_failed or not state.playback_finished:
                return

            try:
                on_complete()
            except Exception:
                LOGGER.exception("TTS completion callback failed.")

        try:
            threading.Thread(
                target=completion_waiter,
                name=f"ZOE-TTS-Completion-{state.generation_id}",
                daemon=True,
            ).start()
        except Exception:
            LOGGER.exception("Could not start completion thread.")

    return state


def speak_async(
    text: str,
    on_complete: Callable[[], None] | None = None,
) -> int | None:
    """
    Start speaking immediately. New speech replaces current speech.
    Returns the generation id, or None if nothing was started.
    """
    state = _start_speech(text, on_complete)
    return state.generation_id if state is not None else None


def speak(
    text: str,
    on_complete: Callable[[], None] | None = None,
    timeout: float | None = None,
) -> bool:
    """
    Blocking speech. Returns True only if the utterance fully played.
    If `timeout` (seconds) is hit, returns False and leaves speech running.
    """
    state = _start_speech(text, on_complete)
    if state is None:
        return False

    if not state.speech_complete_event.wait(timeout):
        return False

    return (
        _is_current(state.generation_id)
        and state.playback_finished
        and not state.cancelled
        and not state.generation_failed
    )


# ============================================================================
# STOP / SHUTDOWN
# ============================================================================


def _invalidate_current() -> None:
    current = _current_generation_id()
    _next_generation_id()
    if current:
        _cancel_state(current)
    _set_speaking(False)
    _clear_generation_queue()
    _clear_audio_queue()


def stop() -> None:
    """Immediately stop playback and invalidate generation."""
    try:
        with _speak_lock:
            _invalidate_current()
    except Exception:
        LOGGER.exception("stop() failed.")


def shutdown() -> None:
    global _generation_worker
    global _playback_worker

    try:
        with _speak_lock:
            _invalidate_current()
            _shutdown_event.set()

        current_thread = threading.current_thread()

        with _workers_lock:
            workers = (_generation_worker, _playback_worker)

        for worker in workers:
            if worker is not None and worker is not current_thread:
                worker.join(timeout=2.0)

        with _workers_lock:
            if _generation_worker is not None and not _generation_worker.is_alive():
                _generation_worker = None
            if _playback_worker is not None and not _playback_worker.is_alive():
                _playback_worker = None
    except Exception:
        LOGGER.exception("shutdown() failed.")


# ============================================================================
# STATUS
# ============================================================================


def get_status() -> dict[str, object]:
    generation_id = _current_generation_id()
    state = _get_state(generation_id)

    stream_alive = _stream_open

    return {
        "speaking": is_speaking(),
        "generation_id": generation_id,
        "generation_queue": _generation_queue.qsize(),
        "audio_queue": _audio_queue.qsize(),
        # Backwards compatibility.
        "audio_process_alive": stream_alive,
        "audio_stream_alive": stream_alive,
        "model_loaded": _client is not None,
        "generation_finished": state.generation_finished if state else False,
        "generation_failed": state.generation_failed if state else False,
        "playback_started": state.playback_started if state else False,
        "playback_finished": state.playback_finished if state else False,
        "error": state.error if state else "",
    }


# ============================================================================
# VOICE FILE GENERATION
# ============================================================================


def generate_voice_file(
    text: str,
    output_path: str | None = None,
) -> str | None:
    """Generate Cartesia speech as a WAV file. Does not play audio."""
    owns_file = output_path is None

    try:
        chunks = _split_sentences(text)
        if not chunks:
            return None

        _get_client()

        if output_path is None:
            fd, output_path = tempfile.mkstemp(prefix="zoe_voice_", suffix=".wav")
            os.close(fd)

        audio_parts: list[bytes] = []

        for chunk in chunks:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    with closing(_iter_pcm(chunk)) as pcm_iter:
                        parts = list(pcm_iter)
                    audio_parts.extend(parts)
                    break
                except Exception as exc:
                    if attempt >= MAX_ATTEMPTS:
                        raise
                    LOGGER.warning("Chunk failed (%s); retrying...", exc)
                    time.sleep(RETRY_DELAY)

        if not audio_parts:
            LOGGER.warning("Cartesia produced no audio.")
            if owns_file:
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            return None

        with wave.open(output_path, "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(b"".join(audio_parts))

        return output_path

    except Exception:
        LOGGER.exception("Failed to generate voice file.")

        if owns_file and output_path:
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ZOE Cartesia TTS")
    parser.add_argument(
        "--text",
        default=(
            "Good evening, Hasin. "
            "ZOE is online. "
            "All systems are operational. "
            "How may I assist you today?"
        ),
    )
    parser.add_argument("--devices", action="store_true")
    parser.add_argument("--file", action="store_true")
    args = parser.parse_args()

    try:
        if args.devices:
            print(list_audio_devices())
        elif args.file:
            print("Generated:", generate_voice_file(args.text))
        else:
            print("ZOE CARTESIA TTS")
            print("Model:", MODEL)
            print("Voice:", VOICE)
            print("Speech completed." if speak(args.text) else "Speech failed or interrupted.")
    except KeyboardInterrupt:
        stop()
    finally:
        shutdown()