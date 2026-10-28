from __future__ import annotations

import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from scipy.signal import butter, resample_poly, sosfilt
from openwakeword.model import Model


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

# Same env var as the STT module. Leave unset to use the system default
# input (set your earbuds as default in OS sound settings), or set it to a
# device index (e.g. "3") or a name fragment (e.g. "airpods", "buds").
MIC_DEVICE_ENV = "ZOE_MIC_DEVICE"

WAKEWORD_MODEL = Path("backend/models/Hey_Zoe.onnx")
WAKEWORD_NAME = "Hey_Zoe"

# Higher = fewer false positives, lower = easier to trigger.
WAKEWORD_THRESHOLD = 0.50
WAKEWORD_REQUIRED_POSITIVES = 2
WAKEWORD_COOLDOWN_SECONDS = 2.0

# openWakeWord works well with ~80 ms chunks at 16 kHz.
WAKE_BLOCK_SIZE = 1280

AUDIO_HIGHPASS_HZ = 80.0
AUDIO_LIMITER_DRIVE = 1.15


# ============================================================
# MIC DEVICE HELPERS
# ============================================================

def list_input_devices() -> list[tuple[int, str, float]]:
    """Return (index, name, default_samplerate) for every input device."""
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
    print(f"[ZOE WAKE] No input device matching {wanted!r}; using system default.")
    return None


# ============================================================
# ZOE WAKEWORD DETECTOR
# ============================================================

class ZoeWakeWord:

    def __init__(
        self,
        on_wake: Optional[Callable[[], None]] = None,
        threshold: float = WAKEWORD_THRESHOLD,
    ):
        self.on_wake = on_wake
        self.threshold = float(threshold)

        self._running = False
        self._runtime_lock = threading.RLock()

        # Microphone
        self._mic_stream: Optional[sd.InputStream] = None
        self._mic_rate = SAMPLE_RATE
        self._mic_resample: Optional[tuple[int, int]] = None
        self._mic_name = ""

        # PortAudio callback -> worker thread (keeps the callback light)
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=200)
        self._worker: Optional[threading.Thread] = None

        # Block accumulator for openWakeWord
        self._wake_audio_buffer = np.zeros(0, dtype=np.float32)

        # Wakeword state
        self._positive_count = 0
        self._last_wake_time = 0.0
        self._latest_score = 0.0

        # High-pass filter (only touched by the worker thread / start / stop)
        self._audio_filter = butter(
            2, AUDIO_HIGHPASS_HZ, btype="highpass", fs=SAMPLE_RATE, output="sos"
        )
        self._audio_filter_state = np.zeros((self._audio_filter.shape[0], 2), dtype=np.float64)
        self._audio_filter_lock = threading.RLock()

        self._wake_model = None
        self._load_wakeword_model()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    def _load_wakeword_model(self) -> None:
        print("[ZOE WAKE] Loading Hey Zoe wakeword model...")
        if not WAKEWORD_MODEL.exists():
            raise FileNotFoundError(f"[ZOE WAKE] Wakeword model not found: {WAKEWORD_MODEL}")
        try:
            self._wake_model = Model(
                wakeword_model_paths=[str(WAKEWORD_MODEL)],
                vad_threshold=0.0,
            )
        except TypeError:
            # Older openWakeWord versions don't accept vad_threshold.
            self._wake_model = Model(wakeword_model_paths=[str(WAKEWORD_MODEL)])
        print("[ZOE WAKE] Hey Zoe model ready.")
        print(f"[ZOE WAKE] Threshold: {self.threshold:.2f}")
        print(f"[ZOE WAKE] Required positives: {WAKEWORD_REQUIRED_POSITIVES}")

    def _reset_audio_filter(self) -> None:
        with self._audio_filter_lock:
            self._audio_filter_state = np.zeros(
                (self._audio_filter.shape[0], 2), dtype=np.float64
            )

    def _reset_wake_model(self) -> None:
        if self._wake_model is None:
            return
        try:
            self._wake_model.reset()
        except Exception:
            pass  # Some openWakeWord versions don't expose reset().

    # --------------------------------------------------------
    # Microphone capture
    # --------------------------------------------------------
    @staticmethod
    def _ratio(native_rate: int) -> tuple[int, int]:
        divisor = math.gcd(SAMPLE_RATE, native_rate)
        return SAMPLE_RATE // divisor, native_rate // divisor  # (up, down)

    def _open_mic(self) -> None:
        device = _resolve_input_device()
        info = sd.query_devices(device, "input")
        self._mic_name = info["name"]
        native_rate = int(info["default_samplerate"])

        # Prefer 16 kHz directly; fall back to the device's native rate and
        # resample (Bluetooth earbuds often insist on 8/16/44.1/48 kHz).
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
                    blocksize=int(rate * WAKE_BLOCK_SIZE / SAMPLE_RATE),
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
            print(f"[ZOE WAKE] Capturing from '{self._mic_name}'{note}.")
            return
        raise RuntimeError(f"Could not open microphone '{self._mic_name}': {last_error}")

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
            print(f"[ZOE WAKE] Stream status: {status}")
        if not self._running:
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

    # --------------------------------------------------------
    # DSP
    # --------------------------------------------------------
    @staticmethod
    def _float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
        audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16)

    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return audio
        with self._audio_filter_lock:
            audio, self._audio_filter_state = sosfilt(
                self._audio_filter, audio, zi=self._audio_filter_state
            )
        audio = np.tanh(audio * AUDIO_LIMITER_DRIVE)  # soft limiter
        return audio.astype(np.float32)

    # --------------------------------------------------------
    # Worker thread
    # --------------------------------------------------------
    def _audio_worker(self) -> None:
        print("[ZOE WAKE] Audio worker started.")
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            if chunk.size:
                self._process_received_audio(self._process_audio(chunk))
        print("[ZOE WAKE] Audio worker stopped.")

    def _process_received_audio(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        self._wake_audio_buffer = np.concatenate((self._wake_audio_buffer, audio))
        while self._wake_audio_buffer.size >= WAKE_BLOCK_SIZE:
            block = self._wake_audio_buffer[:WAKE_BLOCK_SIZE]
            self._wake_audio_buffer = self._wake_audio_buffer[WAKE_BLOCK_SIZE:]
            self._check_wakeword(block)

    # --------------------------------------------------------
    # Wakeword scoring
    # --------------------------------------------------------
    def _get_wake_score(self, prediction) -> float:
        if not prediction:
            return 0.0
        for key in (WAKEWORD_NAME, WAKEWORD_NAME.lower()):
            if key in prediction:
                try:
                    return float(prediction[key])
                except Exception:
                    pass
        try:
            values = list(prediction.values())
            if len(values) == 1:
                return float(values[0])
        except Exception:
            pass
        return 0.0

    def _check_wakeword(self, audio: np.ndarray) -> None:
        if self._wake_model is None or audio.size == 0:
            return

        try:
            prediction = self._wake_model.predict(self._float32_to_pcm16(audio))
        except Exception as e:
            print(f"[ZOE WAKE] Prediction error: {e}")
            return

        score = self._get_wake_score(prediction)
        self._latest_score = score

        if score >= self.threshold:
            self._positive_count += 1
        else:
            self._positive_count = 0

        if self._positive_count < WAKEWORD_REQUIRED_POSITIVES:
            return

        # Reset before the callback so it can't cause duplicate triggers.
        self._positive_count = 0

        now = time.monotonic()
        if now - self._last_wake_time < WAKEWORD_COOLDOWN_SECONDS:
            return
        self._last_wake_time = now

        print(f"[ZOE WAKE] Hey Zoe detected (score={score:.3f})")
        self._reset_wake_model()

        if self.on_wake is not None:
            try:
                self.on_wake()
            except Exception as e:
                print(f"[ZOE WAKE] Wake callback error: {e}")

    # --------------------------------------------------------
    # Start / stop
    # --------------------------------------------------------
    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def start(self) -> "ZoeWakeWord":
        with self._runtime_lock:
            if self._running:
                return self

            self._clear_queue(self._audio_queue)
            self._wake_audio_buffer = np.zeros(0, dtype=np.float32)
            self._positive_count = 0
            self._latest_score = 0.0
            self._reset_audio_filter()
            self._reset_wake_model()

            self._running = True
            try:
                self._open_mic()
            except Exception as exc:
                self._running = False
                print(f"[ZOE WAKE] {exc}")
                print("[ZOE WAKE] Available input devices:")
                for index, name, rate in list_input_devices():
                    print(f"    [{index}] {name} ({rate:.0f} Hz)")
                raise

            self._worker = threading.Thread(
                target=self._audio_worker, name="ZOE-Wake-Audio", daemon=True
            )
            self._worker.start()

        print("[ZOE WAKE] Wakeword listener started.")
        print("[ZOE WAKE] Listening for: Hey Zoe")
        return self

    def stop(self) -> None:
        with self._runtime_lock:
            if not self._running:
                return
            print("[ZOE WAKE] Stopping wakeword listener...")
            self._running = False
            self._close_mic()
            try:
                self._audio_queue.put_nowait(None)
            except queue.Full:
                self._clear_queue(self._audio_queue)
                try:
                    self._audio_queue.put_nowait(None)
                except queue.Full:
                    pass
            worker = self._worker

        if worker and worker is not threading.current_thread():
            worker.join(timeout=3.0)
            if worker.is_alive():
                print("[ZOE WAKE] WARNING: audio worker did not stop within 3 seconds.")

        with self._runtime_lock:
            self._worker = None
            self._positive_count = 0
            self._latest_score = 0.0
            self._wake_audio_buffer = np.zeros(0, dtype=np.float32)
            self._clear_queue(self._audio_queue)
            self._reset_audio_filter()
            self._reset_wake_model()
        print("[ZOE WAKE] Wakeword listener stopped.")

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_mic_active(self) -> bool:
        stream = self._mic_stream
        try:
            return bool(stream is not None and stream.active)
        except Exception:
            return False

    @property
    def is_esp32_connected(self) -> bool:
        # Backwards-compat shim for callers still using the old name.
        return self.is_mic_active

    @property
    def latest_score(self) -> float:
        return self._latest_score


# ============================================================
# SIMPLE CLI TEST
# ============================================================

def main():
    def on_wake():
        print("\n========================================")
        print("       HEY ZOE DETECTED")
        print("========================================\n")

    print("Input devices:")
    for index, name, rate in list_input_devices():
        print(f"  [{index}] {name} ({rate:.0f} Hz)")

    wakeword = ZoeWakeWord(on_wake=on_wake, threshold=WAKEWORD_THRESHOLD)
    try:
        wakeword.start()
        print("[ZOE WAKE] Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[ZOE WAKE] Interrupted.")
    finally:
        wakeword.stop()


if __name__ == "__main__":
    main()