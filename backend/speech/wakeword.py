
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
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

MIC_DEVICE_ENV = "ZOE_MIC_DEVICE"
MODEL_PATH_ENV = "ZOE_WAKEWORD_MODEL"

WAKEWORD_NAME = "Hey_Zoe"
WAKEWORD_MODEL_RELATIVE = Path("backend/models/Hey_Zoe.onnx")

# Higher = fewer false positives, lower = easier to trigger.
WAKEWORD_THRESHOLD = 0.50
WAKEWORD_REQUIRED_POSITIVES = 2
WAKEWORD_COOLDOWN_SECONDS = 2.0

# openWakeWord works well with ~80 ms chunks at 16 kHz.
WAKE_BLOCK_SIZE = 1280

AUDIO_HIGHPASS_HZ = 80.0
AUDIO_LIMITER_DRIVE = 1.15

# On Linux we want to go through the sound server (PipeWire via its ALSA/Pulse
# shims), not a raw hw: device. Raw devices are exclusive, can't be shared with
# the TTS output, and usually only expose the card's native format.
PREFERRED_DEFAULT_DEVICES = ("pipewire", "pulse", "default")

# Hot-plug: how often to check for added/removed input devices.
# Each poll spawns a short-lived Python process (plus pactl); fine on a PC,
# use ~5s on a Raspberry Pi.
MIC_POLL_SECONDS = 3.0


# ============================================================
# PATH / DEVICE HELPERS
# ============================================================

def _resolve_model_path() -> Path:
    override = os.getenv(MODEL_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    candidates = [
        Path.cwd() / WAKEWORD_MODEL_RELATIVE,
        Path(__file__).resolve().parent / WAKEWORD_MODEL_RELATIVE,
        Path(__file__).resolve().parent.parent / WAKEWORD_MODEL_RELATIVE,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def list_input_devices() -> list[tuple[int, str, float]]:
    """Return (index, name, default_samplerate) for every input device."""
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            devices.append((index, info["name"], float(info["default_samplerate"])))
    return devices


def _resolve_input_device() -> Optional[int]:
    """Pick the input device. None means PortAudio's default."""
    wanted = os.getenv(MIC_DEVICE_ENV, "").strip()

    if wanted:
        if wanted.isdigit():
            return int(wanted)
        wanted_lower = wanted.lower()
        for index, name, _rate in list_input_devices():
            if wanted_lower in name.lower():
                return index
        print(f"[ZOE WAKE] No input device matching {wanted!r}; falling back.")

    # No (usable) override: prefer the sound-server devices.
    devices = list_input_devices()
    for preferred in PREFERRED_DEFAULT_DEVICES:
        for index, name, _rate in devices:
            if name.lower() == preferred:
                return index
    return None


# ------------------------------------------------------------
# Hot-plug support
# ------------------------------------------------------------
_PROBE_CODE = (
    "import sounddevice as sd, json\n"
    "devs = [d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]\n"
    "print(json.dumps(devs))\n"
)


def _run_text(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode:
            return None
        return result.stdout.strip()
    except Exception:
        return None


def _probe_input_devices() -> Optional[tuple]:
    """Signature of the real input sources. Prefers the sound server (pactl),
    which is what changes when earbuds connect; PortAudio's own list is noisy
    and only shows virtual devices under PipeWire."""
    if shutil.which("pactl"):
        out = _run_text(["pactl", "list", "short", "sources"])
        if out is not None:
            names = []
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) > 1 and not parts[1].endswith(".monitor"):
                    names.append(parts[1])
            default = _run_text(["pactl", "get-default-source"]) or ""
            return (tuple(sorted(names)), default)
    # Fallback (no pactl): fresh-PortAudio snapshot in a subprocess.
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode:
            return None
        return (tuple(json.loads(result.stdout.strip().splitlines()[-1])), "")
    except Exception:
        return None


def _reset_portaudio() -> None:
    """Force PortAudio to re-enumerate devices. Only safe with no open streams."""
    try:
        sd._terminate()
    except Exception:
        pass
    sd._initialize()


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
        # Anything else holding a PortAudio stream (e.g. TTS playback) must
        # close before a reset and reopen after, because Pa_Terminate kills
        # every open stream in the process.
        self.on_audio_reset_begin: Optional[Callable[[], None]] = None
        self.on_audio_reset_end: Optional[Callable[[], None]] = None

        self._running = False
        self._runtime_lock = threading.RLock()

        # Microphone
        self._mic_stream: Optional[sd.InputStream] = None
        self._mic_rate = SAMPLE_RATE
        self._mic_resample: Optional[tuple[int, int]] = None
        self._mic_name = ""
        self._mic_lock = threading.RLock()
        self._mic_stop = threading.Event()
        self._mic_watch_thread: Optional[threading.Thread] = None
        self._mic_signature: Optional[tuple] = None

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
    def _build_model(self, model_path: Path) -> Model:
        # The default framework on Linux is tflite, which would go looking for
        # a .tflite file. Our model is ONNX, so say so explicitly.
        kwargs = dict(
            wakeword_model_paths=[str(model_path)],
            inference_framework="onnx",
            vad_threshold=0.0,
        )
        try:
            return Model(**kwargs)
        except TypeError:
            # Older openWakeWord versions don't accept these kwargs.
            kwargs.pop("vad_threshold", None)
            try:
                return Model(**kwargs)
            except TypeError:
                kwargs.pop("inference_framework", None)
                return Model(**kwargs)

    def _load_wakeword_model(self) -> None:
        print("[ZOE WAKE] Loading Hey Zoe wakeword model...")
        model_path = _resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"[ZOE WAKE] Wakeword model not found: {model_path}")

        try:
            self._wake_model = self._build_model(model_path)
        except Exception as first_error:
            # openWakeWord's shared melspectrogram/embedding models are not
            # bundled in the pip package; they download on first use.
            print(f"[ZOE WAKE] Model load failed ({first_error}); fetching feature models...")
            try:
                import openwakeword.utils as oww_utils
                oww_utils.download_models()
            except Exception as dl_error:
                raise RuntimeError(
                    f"Could not load or download openWakeWord models: {dl_error}"
                ) from first_error
            self._wake_model = self._build_model(model_path)

        print("[ZOE WAKE] Hey Zoe model ready.")
        print(f"[ZOE WAKE] Model: {model_path}")
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

        # Prefer 16 kHz directly (PipeWire/Pulse resample for us); fall back to
        # the device's native rate and resample ourselves.
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
        # Runs on PortAudio's thread: copy and enqueue only. No DSP here.
        if status:
            print(f"[ZOE WAKE] Stream status: {status}")
        if not self._running:
            return
        chunk = np.array(indata[:, 0], dtype=np.float32, copy=True)
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
    # Mic hot-plug watcher
    # --------------------------------------------------------
    def _reset_audio_state(self) -> None:
        self._clear_queue(self._audio_queue)
        self._wake_audio_buffer = np.zeros(0, dtype=np.float32)
        self._positive_count = 0
        self._latest_score = 0.0
        self._reset_audio_filter()
        self._reset_wake_model()

    def _reconnect_mic(self, new_signature: Optional[tuple], reason: str = "") -> None:
        with self._mic_lock:
            if not self._running:
                return
            print(f"[ZOE WAKE] Reconnecting: {reason}")
            self._close_mic()
            self._reset_audio_state()
            try:
                # Soft path: reopen on the sound server's current default.
                # Does not touch other streams (TTS keeps playing).
                self._open_mic()
            except Exception as soft_exc:
                print(f"[ZOE WAKE] Soft reopen failed ({soft_exc}); resetting PortAudio.")
                if self.on_audio_reset_begin:
                    try:
                        self.on_audio_reset_begin()
                    except Exception as exc:
                        print(f"[ZOE WAKE] Reset-begin hook error: {exc}")
                try:
                    _reset_portaudio()
                    self._open_mic()
                except Exception as exc:
                    print(f"[ZOE WAKE] Reconnect failed (will retry): {exc}")
                finally:
                    if self.on_audio_reset_end:
                        try:
                            self.on_audio_reset_end()
                        except Exception as exc:
                            print(f"[ZOE WAKE] Reset-end hook error: {exc}")
            if new_signature is not None:
                self._mic_signature = new_signature

    def _mic_watcher(self) -> None:
        print("[ZOE WAKE] Device watcher started.")
        while not self._mic_stop.wait(MIC_POLL_SECONDS):
            if not self._running:
                break
            dead = not self.is_mic_active
            signature = _probe_input_devices()
            if signature is None:
                if dead:
                    self._reconnect_mic(None, "stream dead, probe failed")
                continue
            if self._mic_signature is None:
                self._mic_signature = signature  # first baseline
                if not dead:
                    continue
            if dead:
                self._reconnect_mic(signature, "stream not active")
            elif signature != self._mic_signature:
                self._reconnect_mic(signature, f"{self._mic_signature} -> {signature}")
        print("[ZOE WAKE] Device watcher stopped.")

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
        if self._mic_resample is not None:
            up, down = self._mic_resample
            audio = resample_poly(audio, up, down).astype(np.float32)
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

            self._reset_audio_state()
            self._mic_signature = None

            self._running = True
            try:
                with self._mic_lock:
                    self._open_mic()
            except Exception as exc:
                # Don't die: the watcher keeps retrying until a mic shows up.
                print(f"[ZOE WAKE] No usable microphone yet: {exc}")
                try:
                    devices = list_input_devices()
                except Exception:
                    devices = []
                print("[ZOE WAKE] Available input devices:")
                for index, name, rate in devices:
                    print(f"    [{index}] {name} ({rate:.0f} Hz)")
                print("[ZOE WAKE] Will keep retrying in the background.")

            self._worker = threading.Thread(
                target=self._audio_worker, name="ZOE-Wake-Audio", daemon=True
            )
            self._worker.start()

            self._mic_stop.clear()
            self._mic_watch_thread = threading.Thread(
                target=self._mic_watcher, name="ZOE-Wake-MicWatch", daemon=True
            )
            self._mic_watch_thread.start()

        print("[ZOE WAKE] Wakeword listener started.")
        print("[ZOE WAKE] Listening for: Hey Zoe")
        return self

    def stop(self) -> None:
        with self._runtime_lock:
            if not self._running:
                return
            print("[ZOE WAKE] Stopping wakeword listener...")
            self._running = False
            self._mic_stop.set()
            # Taking _mic_lock lets a reconnect in progress finish first, so the
            # watcher can't reopen the mic after we close it.
            with self._mic_lock:
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
            watcher = self._mic_watch_thread

        for thread in (worker, watcher):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    print(f"[ZOE WAKE] WARNING: {thread.name} did not stop within 3 seconds.")

        with self._runtime_lock:
            self._worker = None
            self._mic_watch_thread = None
            self._reset_audio_state()
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
    try:
        for index, name, rate in list_input_devices():
            print(f"  [{index}] {name} ({rate:.0f} Hz)")
    except Exception as exc:
        print(f"  (could not list devices: {exc})")

    wakeword = ZoeWakeWord(on_wake=on_wake, threshold=WAKEWORD_THRESHOLD)

    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        print(f"\n[ZOE WAKE] Received signal {signum}.")
        stop_event.set()

    # SIGTERM matters when this runs under systemd.
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        wakeword.start()
        print("[ZOE WAKE] Press Ctrl+C to stop.")
        while not stop_event.wait(1.0):
            pass
    finally:
        wakeword.stop()


if __name__ == "__main__":
    sys.exit(main())