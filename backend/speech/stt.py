from __future__ import annotations

import io
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import torch
import torch.nn.functional as F
from scipy.signal import butter, resample_poly, sosfilt
from dotenv import load_dotenv
from groq import Groq
from silero_vad import load_silero_vad
from speechbrain.inference.speaker import EncoderClassifier

load_dotenv()

# ============================================================
# Configuration
# ============================================================
GROQ_MODEL = "whisper-large-v3-turbo"  # Compare with whisper-large-v3 after audio is clean.
GROQ_LANGUAGE = "en"
GROQ_TEMPERATURE = 0.0
GROQ_KEYS = [os.getenv("GROQ_API_KEY2"), os.getenv("GROQ_API_KEY3")]
GROQ_KEYS = [key for key in GROQ_KEYS if key]

SAMPLE_RATE = 16000
PCM_BYTES_PER_SAMPLE = 2

# Microphone input. Leave ZOE_MIC_DEVICE unset to use the system default input
# (set your earbuds as the default input in OS sound settings), or set it to a
# device index (e.g. "3") or a name fragment (e.g. "airpods", "buds").
MIC_DEVICE_ENV = "ZOE_MIC_DEVICE"
MIC_BLOCK_SECONDS = 0.032  # 512 samples at 16 kHz, matches the VAD window.

# One host-side high-pass filter.
AUDIO_HIGHPASS_HZ = 70

# Speech segmentation: more pre-roll and more trailing silence preserve word edges.
VAD_SPEECH_THRESHOLD = 0.45
VAD_WINDOW_SAMPLES = 512
MIN_SPEECH_SECONDS = 0.20
END_SILENCE_SECONDS = 0.85
PRE_ROLL_SECONDS = 0.35
MAX_UTTERANCE_SECONDS = 30.0

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
VOICEPRINT_FILE = Path("backend/data/hasin_voiceprint.pt")
SPEAKER_THRESHOLD = 0.45
SPEAKER_MIN_SECONDS = 0.30
SPEAKER_WINDOW_SECONDS = 0.75
SPEAKER_CHECK_INTERVAL = 0.20
BARGE_IN_COOLDOWN_SECONDS = 0.35
MAX_CONSECUTIVE_WORD_REPEAT = 2

# Partial transcription is intentionally disabled for this diagnostic version.
# This avoids duplicate realtime API requests competing with final transcription.
ENABLE_PARTIAL_TRANSCRIPTION = False


@dataclass
class STTResult:
    text: str
    duration: float
    interrupted_tts: bool = False


_groq_lock = threading.RLock()
_groq_clients: dict[str, Groq] = {}
_groq_key_index = 0


def _get_groq_client(api_key: str) -> Groq:
    with _groq_lock:
        if api_key not in _groq_clients:
            _groq_clients[api_key] = Groq(api_key=api_key)
        return _groq_clients[api_key]


if not GROQ_KEYS:
    print("[ZOE GROQ] WARNING: no Groq API keys configured (GROQ_API_KEY2 / GROQ_API_KEY3).")
else:
    print(f"[ZOE GROQ] {len(GROQ_KEYS)} API key(s) configured; model: {GROQ_MODEL}")


def list_input_devices() -> list[tuple[int, str, float]]:
    """Return (index, name, default_samplerate) for every device with input channels."""
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            devices.append((index, info["name"], float(info["default_samplerate"])))
    return devices


def _resolve_input_device() -> Optional[int]:
    """Pick the input device from ZOE_MIC_DEVICE; None means system default."""
    wanted = os.getenv(MIC_DEVICE_ENV, "").strip()
    if not wanted:
        return None
    if wanted.isdigit():
        return int(wanted)
    wanted_lower = wanted.lower()
    for index, name, _rate in list_input_devices():
        if wanted_lower in name.lower():
            return index
    print(f"[ZOE MIC] No input device matching {wanted!r}; falling back to system default.")
    return None


class ZoeSTT:
    def __init__(
        self,
        model: str = GROQ_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
        on_final: Optional[Callable[[str], None]] = None,
        on_interruption: Optional[Callable[[], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.on_final = on_final
        self.on_interruption = on_interruption
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_partial = on_partial
        # Raw audio tap: receives every filtered float32 mono 16 kHz chunk.
        self.on_audio: Optional[Callable[[np.ndarray], None]] = None

        self._running = False
        self._runtime_lock = threading.RLock()
        self._muted = False
        self._mute_lock = threading.RLock()

        # Microphone stream
        self._mic_stream: Optional[sd.InputStream] = None
        self._mic_rate = SAMPLE_RATE
        self._mic_resample: Optional[tuple[int, int]] = None
        self._mic_name = ""

        # Queue now carries float32 mono chunks already at SAMPLE_RATE.
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=200)
        self._final_queue: queue.Queue[Optional[tuple[np.ndarray, float, bool, int, int]]] = queue.Queue(maxsize=8)
        self._audio_thread: Optional[threading.Thread] = None
        self._final_thread: Optional[threading.Thread] = None

        self._filter_sos = butter(2, AUDIO_HIGHPASS_HZ, btype="highpass", fs=SAMPLE_RATE, output="sos")
        self._filter_state = np.zeros((self._filter_sos.shape[0], 2), dtype=np.float64)
        self._filter_lock = threading.RLock()

        self._vad_model = None
        self._vad_buffer = np.zeros(0, dtype=np.float32)
        self._pre_roll: list[np.ndarray] = []
        self._utterance_chunks: list[np.ndarray] = []
        self._utterance_samples = 0
        self._in_speech = False
        self._speech_lock = threading.RLock()
        self._speech_generation = 0
        # Changes only on mute/stop/reset, NOT on each new utterance.
        # A new utterance therefore cannot invalidate a previous final request.
        self._cancel_generation = 0
        self._speech_interrupted_tts = False
        self._last_voice_at = 0.0

        self._speaking = False
        self._speaking_lock = threading.RLock()
        self._interruption_sent = False
        self._last_tts_stop_time = 0.0

        self._speaker_model = None
        self._hasin_voiceprint = None
        self._speaker_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
        self._speaker_thread: Optional[threading.Thread] = None
        self._speaker_worker_running = False
        self._speaker_lock = threading.RLock()
        self._speaker_buffer = np.zeros(0, dtype=np.float32)
        self._last_speaker_check = 0.0
        self._speaker_positive_count = 0
        self._latest_speaker_score = 0.0

        self._load_vad()
        self._load_speaker_verification()

    def _load_vad(self) -> None:
        try:
            print("[ZOE VAD] Loading Silero VAD...")
            self._vad_model = load_silero_vad()
            print("[ZOE VAD] Silero VAD ready.")
        except Exception as exc:
            print(f"[ZOE VAD] Failed to load: {exc}")
            self._vad_model = None

    def _load_speaker_verification(self) -> None:
        try:
            if not VOICEPRINT_FILE.is_file():
                raise FileNotFoundError(f"Voiceprint not found: {VOICEPRINT_FILE}")
            print("[ZOE SPEAKER] Loading ECAPA-TDNN...")
            self._speaker_model = EncoderClassifier.from_hparams(
                source=SPEAKER_MODEL,
                savedir="backend/models/ecapa",
            )
            saved = torch.load(VOICEPRINT_FILE, map_location="cpu", weights_only=False)
            if not isinstance(saved, dict) or "embedding" not in saved:
                raise ValueError("Voiceprint must be a dictionary containing 'embedding'.")
            self._hasin_voiceprint = F.normalize(saved["embedding"].detach().float(), dim=0)
            print(f"[ZOE SPEAKER] ECAPA ready; threshold={SPEAKER_THRESHOLD:.2f}")
        except Exception as exc:
            print(f"[ZOE SPEAKER] Speaker verification unavailable: {exc}")
            self._speaker_model = None
            self._hasin_voiceprint = None

    # ------------------------------------------------------------
    # Microphone capture (replaces the ESP32 TCP bridge)
    # ------------------------------------------------------------
    def _open_mic(self) -> None:
        device = _resolve_input_device()
        info = sd.query_devices(device, "input")
        self._mic_name = info["name"]
        native_rate = int(info["default_samplerate"])

        # Prefer opening directly at 16 kHz; fall back to the device's native
        # rate and resample (Bluetooth earbuds in headset mode often insist on
        # 8/16/44.1/48 kHz depending on driver).
        attempts = [(SAMPLE_RATE, None)]
        if native_rate != SAMPLE_RATE:
            attempts.append((native_rate, self._ratio(native_rate)))

        last_error: Optional[Exception] = None
        for rate, ratio in attempts:
            try:
                stream = sd.InputStream(
                    device=device,
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    blocksize=int(rate * MIC_BLOCK_SECONDS),
                    callback=self._mic_callback,
                )
                stream.start()
            except Exception as exc:
                last_error = exc
                continue
            self._mic_stream = stream
            self._mic_rate = rate
            self._mic_resample = ratio
            note = "" if ratio is None else f" (resampling {rate} -> {SAMPLE_RATE} Hz)"
            print(f"[ZOE MIC] Capturing from '{self._mic_name}'{note}.")
            return
        raise RuntimeError(f"Could not open microphone '{self._mic_name}': {last_error}")

    @staticmethod
    def _ratio(native_rate: int) -> tuple[int, int]:
        divisor = math.gcd(SAMPLE_RATE, native_rate)
        return SAMPLE_RATE // divisor, native_rate // divisor  # (up, down)

    def _close_mic(self) -> None:
        stream, self._mic_stream = self._mic_stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _mic_callback(self, indata, frames, time_info, status) -> None:
        # Runs on PortAudio's thread: keep it light and never block.
        if status:
            print(f"[ZOE MIC] Stream status: {status}")
        if not self._running or self.is_muted():
            return
        chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
        if self._mic_resample is not None:
            up, down = self._mic_resample
            chunk = resample_poly(chunk, up, down).astype(np.float32)
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def _process_input_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if not audio.size:
            return audio
        with self._filter_lock:
            audio, self._filter_state = sosfilt(self._filter_sos, audio, zi=self._filter_state)
        # Do not add a second gate or blockwise AGC: preserve quiet consonants.
        return np.clip(audio, -0.98, 0.98).astype(np.float32)

    @staticmethod
    def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
        pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        wav.write(buffer, SAMPLE_RATE, pcm)
        return buffer.getvalue()

    def _transcribe_with_groq(self, audio: np.ndarray) -> Optional[str]:
        global _groq_key_index
        if not GROQ_KEYS:
            print("[ZOE GROQ] No API keys available.")
            return None
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / SAMPLE_RATE
        if not audio.size or duration < 0.05:
            return None
        audio_bytes = self._audio_to_wav_bytes(audio)
        with _groq_lock:
            start_index = _groq_key_index % len(GROQ_KEYS)
        for attempt in range(len(GROQ_KEYS)):
            index = (start_index + attempt) % len(GROQ_KEYS)
            try:
                response = _get_groq_client(GROQ_KEYS[index]).audio.transcriptions.create(
                    file=("zoe_audio.wav", audio_bytes),
                    model=self.model_name,
                    language=GROQ_LANGUAGE,
                    temperature=GROQ_TEMPERATURE,
                    response_format="json",
                )
                text = self._clean_text(getattr(response, "text", ""))
                with _groq_lock:
                    _groq_key_index = index
                return text or None
            except Exception as exc:
                print(f"[ZOE GROQ] Key {index + 1} failed: {exc}")
                if attempt + 1 < len(GROQ_KEYS):
                    print(f"[ZOE GROQ] Trying key {(index + 1) % len(GROQ_KEYS) + 1}...")
        print("[ZOE GROQ] All configured keys failed.")
        return None

    def _audio_worker(self) -> None:
        print("[ZOE STT] Audio worker started.")
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            if chunk.size:
                processed = self._process_input_audio(chunk)
                if self.on_audio:
                    try:
                        self.on_audio(processed)
                    except Exception:
                        pass
                self._process_audio(processed)
        print("[ZOE STT] Audio worker stopped.")

    def _detect_speech(self, chunk: np.ndarray) -> bool:
        if self._vad_model is None:
            return False
        self._vad_buffer = np.concatenate((self._vad_buffer, chunk))
        detected = False
        while len(self._vad_buffer) >= VAD_WINDOW_SAMPLES:
            window = self._vad_buffer[:VAD_WINDOW_SAMPLES]
            self._vad_buffer = self._vad_buffer[VAD_WINDOW_SAMPLES:]
            try:
                with torch.no_grad():
                    probability = float(self._vad_model(torch.from_numpy(window.copy()), SAMPLE_RATE).item())
                detected = detected or probability >= VAD_SPEECH_THRESHOLD
            except Exception as exc:
                print(f"[ZOE VAD] Inference error: {exc}")
        return detected

    def _process_audio(self, chunk: np.ndarray) -> None:
        if not chunk.size or self.is_muted():
            return
        is_speech = self._detect_speech(chunk)
        now = time.monotonic()
        if is_speech:
            if not self._in_speech:
                self._start_speech()
            self._last_voice_at = now
            self._utterance_chunks.append(chunk.copy())
            self._utterance_samples += len(chunk)
            self._feed_speaker_detector(chunk)
            if self._utterance_samples / SAMPLE_RATE >= MAX_UTTERANCE_SECONDS:
                print("[ZOE VAD] Maximum utterance length reached.")
                self._finish_speech()
            self._update_pre_roll(chunk)
            return

        if self._in_speech:
            # Keep trailing silence in the audio sent to Whisper. It helps preserve
            # the last phoneme and gives the model a natural utterance boundary.
            self._utterance_chunks.append(chunk.copy())
            self._utterance_samples += len(chunk)
            if now - self._last_voice_at >= END_SILENCE_SECONDS:
                self._finish_speech()
        self._update_pre_roll(chunk)

    def _update_pre_roll(self, chunk: np.ndarray) -> None:
        self._pre_roll.append(chunk.copy())
        total = sum(len(part) for part in self._pre_roll)
        max_samples = int(PRE_ROLL_SECONDS * SAMPLE_RATE)
        while self._pre_roll and total > max_samples:
            total -= len(self._pre_roll.pop(0))

    def _start_speech(self) -> None:
        with self._speech_lock:
            if self._in_speech:
                return
            self._in_speech = True
            self._speech_generation += 1
            generation = self._speech_generation
            self._speech_interrupted_tts = self.is_speaking()
        self._utterance_chunks = [part.copy() for part in self._pre_roll]
        self._utterance_samples = sum(len(part) for part in self._utterance_chunks)
        print(f"\n[ZOE VAD] Speech started (utterance={generation}).")
        if self.on_speech_start:
            try:
                self.on_speech_start()
            except Exception as exc:
                print(f"[ZOE STT] Speech-start callback error: {exc}")

    def _finish_speech(self) -> None:
        with self._speech_lock:
            if not self._in_speech:
                return
            self._in_speech = False
            self._speech_generation += 1
            generation = self._speech_generation
            interrupted = self._speech_interrupted_tts
            self._speech_interrupted_tts = False
            chunks = list(self._utterance_chunks)
            cancel_generation = self._cancel_generation
        if self.on_speech_end:
            try:
                self.on_speech_end()
            except Exception as exc:
                print(f"[ZOE STT] Speech-end callback error: {exc}")
        self._reset_utterance()
        if not chunks:
            return
        audio = np.concatenate(chunks).astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        print(f"\n[ZOE STT] Utterance complete: {duration:.2f}s; final transcription queued.")
        if duration < MIN_SPEECH_SECONDS:
            print("[ZOE STT] Utterance too short.")
            return
        item = (audio, duration, interrupted, generation, cancel_generation)
        try:
            self._final_queue.put_nowait(item)
        except queue.Full:
            # Drop oldest waiting utterance, never block the audio-capture thread.
            try:
                self._final_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._final_queue.put_nowait(item)
            except queue.Full:
                print("[ZOE STT] Final queue full; utterance dropped.")

    def _final_worker(self) -> None:
        print("[ZOE STT] Final transcription worker started.")
        while self._running:
            try:
                item = self._final_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            self._transcribe_final(*item)
        print("[ZOE STT] Final transcription worker stopped.")

    def _transcribe_final(
        self,
        audio: np.ndarray,
        duration: float,
        interrupted: bool,
        generation: int,
        cancel_generation: int,
    ) -> None:
        # Only explicit mute/stop invalidates an in-flight request. Starting a new
        # utterance no longer discards this completed utterance's result.
        if self._is_cancelled(cancel_generation):
            print("[ZOE STT] Final cancelled by mute/stop.")
            return
        if self._speaker_model is None or self._hasin_voiceprint is None:
            print("[ZOE STT] Final rejected: speaker verification unavailable.")
            return
        print("[ZOE SPEAKER] Verifying complete utterance...")
        is_hasin, score = self._verify_speaker(audio)
        if not is_hasin:
            print(f"[ZOE STT] Speaker rejected (score={score:.4f}).")
            return
        print(f"[ZOE SPEAKER] Speaker verified (score={score:.4f}).")
        if self._is_cancelled(cancel_generation):
            print("[ZOE STT] Final cancelled by mute/stop during verification.")
            return

        print(f"[ZOE STT] Groq Whisper {self.model_name} transcribing...")
        text = self._transcribe_with_groq(audio)
        if not text:
            print("[ZOE STT] Groq returned no transcription.")
            return
        text = self._clean_text(self._remove_repeated_text(text))
        if not text or self._looks_like_hallucination(text, duration):
            print(f"[ZOE STT] Ignored likely hallucination: {text!r}")
            return
        if self._is_cancelled(cancel_generation):
            print("[ZOE STT] Final cancelled by mute/stop after transcription.")
            return
        result = STTResult(text, duration, interrupted)
        print(f"[ZOE STT] Final (utterance={generation}): {result.text}")
        if result.interrupted_tts:
            print("[ZOE STT] Barge-in command accepted.")
        if self.on_final:
            try:
                self.on_final(result.text)
            except Exception as exc:
                print(f"[ZOE STT] Final callback error: {exc}")

    def _is_cancelled(self, token: int) -> bool:
        with self._speech_lock:
            return (not self._running) or token != self._cancel_generation

    def _speaker_score(self, audio: np.ndarray) -> float:
        if self._speaker_model is None or self._hasin_voiceprint is None or len(audio) / SAMPLE_RATE < SPEAKER_MIN_SECONDS:
            return 0.0
        try:
            audio = np.asarray(audio, dtype=np.float32)
            audio = audio - np.mean(audio)
            tensor = torch.from_numpy(audio.copy()).unsqueeze(0)
            with torch.no_grad():
                embedding = self._speaker_model.encode_batch(tensor).squeeze().float()
            embedding = F.normalize(embedding, dim=0)
            return float(torch.dot(self._hasin_voiceprint, embedding).item())
        except Exception as exc:
            print(f"[ZOE SPEAKER] Score error: {exc}")
            return 0.0

    def _verify_speaker(self, audio: np.ndarray) -> tuple[bool, float]:
        if len(audio) / SAMPLE_RATE < SPEAKER_MIN_SECONDS:
            return False, 0.0
        score = self._speaker_score(audio)
        print(f"[ZOE SPEAKER] {'HASIN' if score >= SPEAKER_THRESHOLD else 'OTHER SPEAKER'} score={score:.4f}")
        return score >= SPEAKER_THRESHOLD, score

    def _feed_speaker_detector(self, chunk: np.ndarray) -> None:
        if self._speaker_model is None or self._hasin_voiceprint is None or not self.is_speaking():
            return
        required = int(SPEAKER_WINDOW_SECONDS * SAMPLE_RATE)
        with self._speaker_lock:
            self._speaker_buffer = np.concatenate((self._speaker_buffer, chunk))[-required:]
            if len(self._speaker_buffer) < required:
                return
            now = time.monotonic()
            if now - self._last_speaker_check < SPEAKER_CHECK_INTERVAL:
                return
            self._last_speaker_check = now
            window = self._speaker_buffer.copy()
        try:
            self._speaker_queue.put_nowait(window)
        except queue.Full:
            try:
                self._speaker_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._speaker_queue.put_nowait(window)
            except queue.Full:
                pass

    def _speaker_worker(self) -> None:
        print("[ZOE SPEAKER] Barge-in worker started.")
        while self._running and self._speaker_worker_running:
            try:
                audio = self._speaker_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio is None:
                break
            if not self.is_speaking():
                continue
            score = self._speaker_score(audio)
            with self._speaker_lock:
                self._latest_speaker_score = score
                self._speaker_positive_count = self._speaker_positive_count + 1 if score >= SPEAKER_THRESHOLD else 0
                positives = self._speaker_positive_count
                if positives >= 1:
                    self._speaker_positive_count = 0
            if positives >= 1:
                self._trigger_hasin_barge_in()
        print("[ZOE SPEAKER] Barge-in worker stopped.")

    def _trigger_hasin_barge_in(self) -> None:
        now = time.monotonic()
        with self._speaking_lock:
            if not self._speaking or self._interruption_sent or now - self._last_tts_stop_time < BARGE_IN_COOLDOWN_SECONDS:
                return
            self._interruption_sent = True
            self._speaking = False
            self._last_tts_stop_time = now
        with self._speech_lock:
            self._speech_interrupted_tts = True
        print("\n[ZOE BARGE] Hasin detected; stopping ZOE speech.")
        if self.on_interruption:
            try:
                self.on_interruption()
            except Exception as exc:
                print(f"[ZOE STT] Interruption callback error: {exc}")

    def _reset_speaker_detection(self) -> None:
        with self._speaker_lock:
            self._speaker_buffer = np.zeros(0, dtype=np.float32)
            self._last_speaker_check = 0.0
            self._speaker_positive_count = 0
            self._latest_speaker_score = 0.0
        while True:
            try:
                self._speaker_queue.get_nowait()
            except queue.Empty:
                break

    def start(self) -> "ZoeSTT":
        with self._runtime_lock:
            if self._running:
                return self
            self._running = True
            with self._speech_lock:
                self._speech_generation += 1
                self._cancel_generation += 1
                self._in_speech = False
                self._speech_interrupted_tts = False
            self._reset_utterance()
            self._vad_buffer = np.zeros(0, dtype=np.float32)
            self._pre_roll.clear()
            with self._filter_lock:
                self._filter_state.fill(0)
            self._clear_queue(self._audio_queue)
            self._clear_queue(self._final_queue)
            self._clear_queue(self._speaker_queue)
            try:
                self._open_mic()
            except Exception as exc:
                self._running = False
                print(f"[ZOE MIC] {exc}")
                print("[ZOE MIC] Available input devices:")
                for index, name, rate in list_input_devices():
                    print(f"    [{index}] {name} ({rate:.0f} Hz)")
                raise
            self._audio_thread = threading.Thread(target=self._audio_worker, name="ZOE-STT-Audio", daemon=True)
            self._final_thread = threading.Thread(target=self._final_worker, name="ZOE-STT-Final", daemon=True)
            self._speaker_worker_running = True
            self._speaker_thread = threading.Thread(target=self._speaker_worker, name="ZOE-Speaker", daemon=True)
            for thread in (self._audio_thread, self._final_thread, self._speaker_thread):
                thread.start()
        print("[ZOE STT] Listening: local microphone -> float32 mono 16 kHz -> final Groq Whisper.")
        print("[ZOE STT] Partials disabled; VAD, ECAPA verification and TTS barge-in enabled where available.")
        return self

    def stop(self) -> None:
        with self._runtime_lock:
            if not self._running:
                return
            with self._speech_lock:
                self._running = False
                was_in_speech = self._in_speech
                self._in_speech = False
                self._speech_generation += 1
                self._cancel_generation += 1
                self._speech_interrupted_tts = False
            if was_in_speech and self.on_speech_end:
                try:
                    self.on_speech_end()
                except Exception as exc:
                    print(f"[ZOE STT] Speech-end callback error: {exc}")
            self._close_mic()
            self._speaker_worker_running = False
            for q in (self._audio_queue, self._final_queue, self._speaker_queue):
                try:
                    q.put_nowait(None)
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(None)
                    except queue.Full:
                        pass
            threads = (self._audio_thread, self._final_thread, self._speaker_thread)
        for thread in threads:
            if thread and thread is not threading.current_thread():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    print(f"[ZOE STT] WARNING: {thread.name} did not stop within 3 seconds.")
        with self._runtime_lock:
            self._audio_thread = self._final_thread = self._speaker_thread = None
            self._reset_utterance()
            self._pre_roll.clear()
            self._vad_buffer = np.zeros(0, dtype=np.float32)
            self._reset_speaker_detection()
        print("[ZOE STT] Stopped.")

    def set_muted(self, muted: bool) -> None:
        muted = bool(muted)
        with self._mute_lock:
            if muted == self._muted:
                return
            self._muted = muted
        if muted:
            with self._speech_lock:
                was_speaking = self._in_speech
                self._in_speech = False
                self._speech_generation += 1
                self._cancel_generation += 1
            self._reset_utterance()
            self._pre_roll.clear()
            self._vad_buffer = np.zeros(0, dtype=np.float32)
            self._reset_speaker_detection()
            self._clear_queue(self._audio_queue)
            with self._filter_lock:
                self._filter_state.fill(0)
            if was_speaking and self.on_speech_end:
                try:
                    self.on_speech_end()
                except Exception as exc:
                    print(f"[ZOE STT] Speech-end callback error: {exc}")
            print("[ZOE STT] Microphone muted; in-flight final transcription cancelled.")
        else:
            self._reset_utterance()
            self._pre_roll.clear()
            self._vad_buffer = np.zeros(0, dtype=np.float32)
            self._reset_speaker_detection()
            with self._filter_lock:
                self._filter_state.fill(0)
            print("[ZOE STT] Microphone unmuted.")

    def is_muted(self) -> bool:
        with self._mute_lock:
            return self._muted

    def set_speaking(self, speaking: bool) -> None:
        speaking = bool(speaking)
        with self._speaking_lock:
            previous = self._speaking
            self._speaking = speaking
            if speaking and not previous:
                self._interruption_sent = False
                self._reset_speaker_detection()
                print("[ZOE STT] TTS started; speaker barge-in active.")
            elif not speaking and previous:
                self._last_tts_stop_time = time.monotonic()
                self._reset_speaker_detection()
                print("[ZOE STT] TTS ended.")

    def is_speaking(self) -> bool:
        with self._speaking_lock:
            return self._speaking

    def is_running(self) -> bool:
        with self._runtime_lock:
            return self._running

    def is_user_speaking(self) -> bool:
        with self._speech_lock:
            return self._in_speech

    def is_mic_active(self) -> bool:
        stream = self._mic_stream
        try:
            return bool(stream is not None and stream.active)
        except Exception:
            return False

    def is_esp32_connected(self) -> bool:
        # Backwards-compat shim for callers that still use the old name.
        return self.is_mic_active()

    def get_speaker_score(self) -> float:
        with self._speaker_lock:
            return self._latest_speaker_score

    def get_partial_text(self) -> str:
        return ""  # Partials intentionally disabled in this version.

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(str(text).split()).strip()

    @staticmethod
    def _remove_repeated_text(text: str, max_repeat: int = MAX_CONSECUTIVE_WORD_REPEAT) -> str:
        output: list[str] = []
        for token in text.split():
            normalized = token.strip(".,!?;:").lower()
            if normalized and len(output) >= max_repeat and all(
                previous.strip(".,!?;:").lower() == normalized for previous in output[-max_repeat:]
            ):
                continue
            output.append(token)
        return " ".join(output)

    @staticmethod
    def _looks_like_hallucination(text: str, duration: float) -> bool:
        words = text.lower().split()
        if not words:
            return True
        if duration < 0.7 and len(words) > 8:
            return True
        if len(words) >= 3 and len(set(words)) == 1:
            return True
        if duration < 1.0 and text.lower() in {
            "thank you", "thanks", "thanks for watching", "thank you for watching", "you", "bye"
        }:
            return True
        return False

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _reset_utterance(self) -> None:
        self._utterance_chunks = []
        self._utterance_samples = 0


def transcribe_audio_file(audio_path: str) -> Optional[str]:
    path = Path(audio_path)
    if not path.is_file():
        print(f"[ZOE STT] Audio file not found: {audio_path}")
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[ZOE STT] FFmpeg is required for file transcription but was not found.")
        return None
    wav_path = None
    try:
        fd, wav_path = tempfile.mkstemp(prefix="zoe_stt_", suffix=".wav")
        os.close(fd)
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120, check=False,
        )
        if result.returncode:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            print(f"[ZOE STT] FFmpeg conversion failed: {error or 'unknown error'}")
            return None
        sample_rate, audio = wav.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            audio = audio.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            audio = audio.astype(np.float32)
        if sample_rate != SAMPLE_RATE or not audio.size:
            print(f"[ZOE STT] Invalid file audio: sample_rate={sample_rate}, samples={audio.size}")
            return None
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_SECONDS:
            print(f"[ZOE STT] Audio too short: {duration:.2f}s")
            return None
        audio -= np.mean(audio)
        if not GROQ_KEYS:
            print("[ZOE GROQ] No API keys available.")
            return None
        # Reuse the module's standard transcription path without constructing an
        # incompletely initialized ZoeSTT instance.
        temporary = object.__new__(ZoeSTT)
        temporary.model_name = GROQ_MODEL
        text = temporary._transcribe_with_groq(audio)
        if not text:
            return None
        text = ZoeSTT._clean_text(ZoeSTT._remove_repeated_text(text))
        if not text or ZoeSTT._looks_like_hallucination(text, duration):
            print(f"[ZOE STT] Ignored file transcription: {text!r}")
            return None
        print(f"[ZOE STT] File transcript: {text}")
        return text
    except subprocess.TimeoutExpired:
        print("[ZOE STT] FFmpeg conversion timed out.")
        return None
    except Exception as exc:
        print(f"[ZOE STT] File transcription failed: {exc}")
        return None
    finally:
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass


_stt: Optional[ZoeSTT] = None
_stt_global_lock = threading.RLock()


def start(
    on_final: Optional[Callable[[str], None]] = None,
    on_interruption: Optional[Callable[[], None]] = None,
    on_speech_start: Optional[Callable[[], None]] = None,
    on_speech_end: Optional[Callable[[], None]] = None,
    on_partial: Optional[Callable[[str], None]] = None,
) -> ZoeSTT:
    global _stt
    with _stt_global_lock:
        if _stt is None:
            _stt = ZoeSTT(
                model=GROQ_MODEL,
                on_final=on_final,
                on_interruption=on_interruption,
                on_speech_start=on_speech_start,
                on_speech_end=on_speech_end,
                on_partial=on_partial,
            )
        else:
            for name, callback in (
                ("on_final", on_final),
                ("on_interruption", on_interruption),
                ("on_speech_start", on_speech_start),
                ("on_speech_end", on_speech_end),
                ("on_partial", on_partial),
            ):
                if callback is not None:
                    setattr(_stt, name, callback)
        return _stt.start()


def stop() -> None:
    with _stt_global_lock:
        if _stt is not None:
            _stt.stop()


def set_speaking(speaking: bool) -> None:
    with _stt_global_lock:
        if _stt is not None:
            _stt.set_speaking(speaking)


def is_speaking() -> bool:
    with _stt_global_lock:
        return _stt.is_speaking() if _stt else False


def set_muted(muted: bool) -> None:
    with _stt_global_lock:
        if _stt is not None:
            _stt.set_muted(muted)


def is_muted() -> bool:
    with _stt_global_lock:
        return _stt.is_muted() if _stt else False


def is_running() -> bool:
    with _stt_global_lock:
        return _stt.is_running() if _stt else False


def get_partial_text() -> str:
    with _stt_global_lock:
        return _stt.get_partial_text() if _stt else ""


def is_user_speaking() -> bool:
    with _stt_global_lock:
        return _stt.is_user_speaking() if _stt else False


def get_speaker_score() -> float:
    with _stt_global_lock:
        return _stt.get_speaker_score() if _stt else 0.0


def is_mic_active() -> bool:
    with _stt_global_lock:
        return _stt.is_mic_active() if _stt else False


def is_esp32_connected() -> bool:
    # Backwards-compat shim: now reports whether the local mic stream is live.
    return is_mic_active()


class MicMonitor:
    """Plays mic audio back to the default output with a hard latency cap."""

    def __init__(self, gain: float = 1.0, max_latency: float = 0.25):
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._buf = np.zeros(0, dtype=np.float32)
        self._max = int(max_latency * SAMPLE_RATE)
        self._gain = gain
        self._stream: Optional[sd.OutputStream] = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        while len(self._buf) < frames:
            try:
                self._buf = np.concatenate((self._buf, self._q.get_nowait()))
            except queue.Empty:
                break
        if len(self._buf) > self._max:  # fell behind: drop old audio, keep latency low
            self._buf = self._buf[-self._max:]
        n = min(frames, len(self._buf))
        outdata[:n, 0] = np.clip(self._buf[:n] * self._gain, -1.0, 1.0)
        outdata[n:, 0] = 0.0
        self._buf = self._buf[n:]

    def feed(self, chunk: np.ndarray) -> None:
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            pass

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=512, latency="low", callback=self._callback,
        )
        self._stream.start()
        print("[ZOE MONITOR] Live mic playback on.")

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


if __name__ == "__main__":
    def final_text(text: str) -> None:
        print(f"\n\n========== FINAL TRANSCRIPT ==========\n{text}\n")

    def interrupted() -> None:
        print("\n[ZOE] User interrupted TTS.")

    def speech_started() -> None:
        print("\n[ZOE] User speech detected.")

    def speech_ended() -> None:
        print("\n[ZOE] User speech ended.")

    print("Input devices:")
    for _index, _name, _rate in list_input_devices():
        print(f"  [{_index}] {_name} ({_rate:.0f} Hz)")

    monitor = MicMonitor(gain=1.0)
    monitor.start()

    stt = start(
        on_final=final_text,
        on_interruption=interrupted,
        on_speech_start=speech_started,
        on_speech_end=speech_ended,
    )
    stt.on_audio = monitor.feed

    print("ZOE local-mic STT | 16 kHz mono | Silero VAD | ECAPA | Groq Whisper")
    print("Listening on the selected microphone. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[ZOE] Shutting down...")
        monitor.stop()
        stop()