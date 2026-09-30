from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, sosfilt
from openwakeword.model import Model

from backend.speech.listening import get_bridge


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

# ------------------------------------------------------------
# Wakeword
# ------------------------------------------------------------

WAKEWORD_MODEL = Path(
    "backend/models/Hey_Zoe.onnx"
)

WAKEWORD_NAME = "Hey_Zoe"

# Detection threshold.
#
# Higher = fewer false positives
# Lower  = easier to trigger
#
# Start around 0.50 and tune from there.
WAKEWORD_THRESHOLD = 0.50

# Number of consecutive positive predictions required.
WAKEWORD_REQUIRED_POSITIVES = 2

# Prevent immediately triggering multiple times.
WAKEWORD_COOLDOWN_SECONDS = 2.0

# openWakeWord works well with approximately
# 80 ms chunks at 16 kHz.
WAKE_BLOCK_SIZE = 1280

# ------------------------------------------------------------
# Audio DSP
# ------------------------------------------------------------

AUDIO_HIGHPASS_HZ = 80.0
AUDIO_LIMITER_DRIVE = 1.15


# ============================================================
# ZOE WAKEWORD DETECTOR
# ============================================================

class ZoeWakeWord:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        on_wake: Optional[Callable[[], None]] = None,
        threshold: float = WAKEWORD_THRESHOLD,
    ):
        self.on_wake = on_wake
        self.threshold = float(threshold)

        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        self._running = False

        # ----------------------------------------------------
        # SHARED ESP32 AUDIO BRIDGE
        # ----------------------------------------------------
        #
        # The bridge owns:
        #
        #   - TCP server
        #   - ESP32 connection
        #   - socket recv()
        #   - reconnect handling
        #   - raw PCM delivery
        #
        # This class only subscribes to incoming audio.
        #

        self._bridge = get_bridge()

        # ----------------------------------------------------
        # PCM BUFFER
        #
        # TCP framing is handled by the bridge, but we still
        # keep a byte buffer here because the bridge may deliver
        # arbitrary-sized byte chunks.
        # ----------------------------------------------------

        self._pcm_buffer = bytearray()

        # ----------------------------------------------------
        # WAKEWORD AUDIO BUFFER
        #
        # Network/audio callbacks do not necessarily arrive in
        # exact openWakeWord block sizes.
        # ----------------------------------------------------

        self._wake_audio_buffer = np.zeros(
            0,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # WAKEWORD STATE
        # ----------------------------------------------------

        self._positive_count = 0
        self._last_wake_time = 0.0
        self._latest_score = 0.0

        # ----------------------------------------------------
        # AUDIO FILTER
        # ----------------------------------------------------

        self._audio_filter = butter(
            2,
            AUDIO_HIGHPASS_HZ,
            btype="highpass",
            fs=SAMPLE_RATE,
            output="sos",
        )

        self._audio_filter_state = np.zeros(
            (
                self._audio_filter.shape[0],
                2,
            ),
            dtype=np.float64,
        )

        self._audio_filter_lock = threading.RLock()

        # ----------------------------------------------------
        # OPENWAKEWORD
        # ----------------------------------------------------

        self._wake_model = None

        self._load_wakeword_model()

    # ========================================================
    # LOAD WAKEWORD MODEL
    # ========================================================

    def _load_wakeword_model(self):
        print(
            "[ZOE WAKE] Loading Hey Zoe wakeword model..."
        )

        if not WAKEWORD_MODEL.exists():
            raise FileNotFoundError(
                "[ZOE WAKE] Wakeword model not found: "
                f"{WAKEWORD_MODEL}"
            )

        try:
            self._wake_model = Model(
                wakeword_model_paths=[
                    str(WAKEWORD_MODEL)
                ],
                vad_threshold=0.0,
            )

        except TypeError:
            # Compatibility fallback for versions where
            # vad_threshold is not accepted.
            self._wake_model = Model(
                wakeword_model_paths=[
                    str(WAKEWORD_MODEL)
                ]
            )

        print(
            "[ZOE WAKE] Hey Zoe model ready."
        )

        print(
            "[ZOE WAKE] "
            f"Threshold: {self.threshold:.2f}"
        )

        print(
            "[ZOE WAKE] "
            f"Required positives: "
            f"{WAKEWORD_REQUIRED_POSITIVES}"
        )

    # ========================================================
    # RESET AUDIO FILTER
    # ========================================================

    def _reset_audio_filter(self):
        with self._audio_filter_lock:
            self._audio_filter_state = np.zeros(
                (
                    self._audio_filter.shape[0],
                    2,
                ),
                dtype=np.float64,
            )

    # ========================================================
    # RESET WAKEWORD MODEL
    # ========================================================

    def _reset_wake_model(self):
        """
        Reset openWakeWord's internal prediction state.

        This prevents audio from immediately before a trigger
        from contributing to the next detection.
        """

        if self._wake_model is None:
            return

        try:
            self._wake_model.reset()

        except Exception:
            # Some openWakeWord versions may not expose reset().
            pass

    # ========================================================
    # PCM16 -> FLOAT32
    # ========================================================

    @staticmethod
    def _pcm16_to_float32(
        data: bytes,
    ) -> np.ndarray:

        if not data:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        # Only complete 16-bit samples.
        sample_count = len(data) // 2

        if sample_count <= 0:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        data = data[
            : sample_count * 2
        ]

        pcm = np.frombuffer(
            data,
            dtype="<i2",
        )

        return (
            pcm.astype(
                np.float32
            )
            / 32768.0
        )

    # ========================================================
    # FLOAT32 -> PCM16
    # ========================================================

    @staticmethod
    def _float32_to_pcm16(
        audio: np.ndarray,
    ) -> np.ndarray:

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        audio = np.clip(
            audio,
            -1.0,
            1.0,
        )

        return (
            audio * 32767.0
        ).astype(
            np.int16
        )

    # ========================================================
    # AUDIO DSP
    # ========================================================

    def _process_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return audio

        # ----------------------------------------------------
        # HIGH-PASS FILTER
        # ----------------------------------------------------

        with self._audio_filter_lock:

            audio, self._audio_filter_state = (
                sosfilt(
                    self._audio_filter,
                    audio,
                    zi=self._audio_filter_state,
                )
            )

        # ----------------------------------------------------
        # SOFT LIMITER
        # ----------------------------------------------------

        audio = np.tanh(
            audio * AUDIO_LIMITER_DRIVE
        )

        return audio.astype(
            np.float32
        )

    # ========================================================
    # RECEIVE AUDIO FROM ESP32 BRIDGE
    # ========================================================

    def _on_audio_bytes(
        self,
        data: bytes,
    ):
        """
        Called by esp32_bridge whenever raw microphone PCM
        bytes arrive.

        The bridge owns the TCP connection. This method only
        consumes the incoming audio stream.
        """

        if not self._running:
            return

        if not data:
            return

        # ----------------------------------------------------
        # BUFFER RAW PCM BYTES
        # ----------------------------------------------------

        self._pcm_buffer.extend(data)

        # ----------------------------------------------------
        # CONSUME COMPLETE PCM16 SAMPLES
        # ----------------------------------------------------

        complete_bytes = (
            len(self._pcm_buffer) // 2
        ) * 2

        if complete_bytes <= 0:
            return

        raw = bytes(
            self._pcm_buffer[
                :complete_bytes
            ]
        )

        del self._pcm_buffer[
            :complete_bytes
        ]

        # ----------------------------------------------------
        # PCM16 -> FLOAT32
        # ----------------------------------------------------

        audio = self._pcm16_to_float32(
            raw
        )

        if audio.size == 0:
            return

        # ----------------------------------------------------
        # AUDIO DSP
        # ----------------------------------------------------

        audio = self._process_audio(
            audio
        )

        # ----------------------------------------------------
        # WAKEWORD PROCESSING
        # ----------------------------------------------------

        self._process_received_audio(
            audio
        )

    # ========================================================
    # FIND WAKEWORD SCORE
    # ========================================================

    def _get_wake_score(
        self,
        prediction,
    ) -> float:

        if not prediction:
            return 0.0

        # ----------------------------------------------------
        # EXPECTED KEY
        # ----------------------------------------------------

        if WAKEWORD_NAME in prediction:

            try:
                return float(
                    prediction[
                        WAKEWORD_NAME
                    ]
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # LOWERCASE KEY
        # ----------------------------------------------------

        lowercase_name = (
            WAKEWORD_NAME.lower()
        )

        if lowercase_name in prediction:

            try:
                return float(
                    prediction[
                        lowercase_name
                    ]
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # SINGLE PREDICTION FALLBACK
        # ----------------------------------------------------

        try:

            values = list(
                prediction.values()
            )

            if len(values) == 1:

                return float(
                    values[0]
                )

        except Exception:
            pass

        return 0.0

    # ========================================================
    # CHECK WAKEWORD
    # ========================================================

    def _check_wakeword(
        self,
        audio: np.ndarray,
    ):

        if self._wake_model is None:
            return

        if audio.size == 0:
            return

        # ----------------------------------------------------
        # FLOAT32 -> PCM16
        #
        # openWakeWord expects PCM audio.
        # ----------------------------------------------------

        pcm16 = self._float32_to_pcm16(
            audio
        )

        try:

            prediction = (
                self._wake_model.predict(
                    pcm16
                )
            )

        except Exception as e:

            print(
                "[ZOE WAKE] "
                f"Prediction error: {e}"
            )

            return

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = self._get_wake_score(
            prediction
        )

        self._latest_score = score

        # ----------------------------------------------------
        # POSITIVE
        # ----------------------------------------------------

        if score >= self.threshold:

            self._positive_count += 1

        else:

            self._positive_count = 0

        # ----------------------------------------------------
        # REQUIRED POSITIVES
        # ----------------------------------------------------

        if (
            self._positive_count
            < WAKEWORD_REQUIRED_POSITIVES
        ):
            return

        # Reset before callback so callback processing cannot
        # accidentally cause duplicate triggers.
        self._positive_count = 0

        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        now = time.monotonic()

        if (
            now - self._last_wake_time
            < WAKEWORD_COOLDOWN_SECONDS
        ):
            return

        self._last_wake_time = now

        print(
            "[ZOE WAKE] "
            f"Hey Zoe detected "
            f"(score={score:.3f})"
        )

        # ----------------------------------------------------
        # RESET MODEL STATE
        # ----------------------------------------------------

        self._reset_wake_model()

        # ----------------------------------------------------
        # CALLBACK
        # ----------------------------------------------------

        if self.on_wake is not None:

            try:

                self.on_wake()

            except Exception as e:

                print(
                    "[ZOE WAKE] "
                    f"Wake callback error: {e}"
                )

    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    def _process_received_audio(
        self,
        audio: np.ndarray,
    ):

        if audio.size == 0:
            return

        # ----------------------------------------------------
        # FEED OPENWAKEWORD IN STABLE BLOCKS
        # ----------------------------------------------------

        self._wake_audio_buffer = np.concatenate(
            (
                self._wake_audio_buffer,
                audio,
            )
        )

        while (
            self._wake_audio_buffer.size
            >= WAKE_BLOCK_SIZE
        ):

            block = (
                self._wake_audio_buffer[
                    :WAKE_BLOCK_SIZE
                ]
            )

            self._wake_audio_buffer = (
                self._wake_audio_buffer[
                    WAKE_BLOCK_SIZE:
                ]
            )

            self._check_wakeword(
                block
            )

    # ========================================================
    # START
    # ========================================================

    def start(self):

        if self._running:
            return

        self._running = True

        # ----------------------------------------------------
        # RESET AUDIO STATE
        # ----------------------------------------------------

        self._pcm_buffer.clear()

        self._wake_audio_buffer = np.zeros(
            0,
            dtype=np.float32,
        )

        self._positive_count = 0
        self._latest_score = 0.0

        self._reset_audio_filter()

        self._reset_wake_model()

        # ----------------------------------------------------
        # SUBSCRIBE TO SHARED ESP32 BRIDGE
        # ----------------------------------------------------

        self._bridge.subscribe(
            self._on_audio_bytes
        )

        print(
            "[ZOE WAKE] "
            "Wakeword listener started."
        )

        print(
            "[ZOE WAKE] "
            "Listening for: Hey Zoe"
        )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        if not self._running:
            return

        print(
            "[ZOE WAKE] "
            "Stopping wakeword listener..."
        )

        self._running = False

        # ----------------------------------------------------
        # UNSUBSCRIBE FROM SHARED BRIDGE
        # ----------------------------------------------------

        self._bridge.unsubscribe(
            self._on_audio_bytes
        )

        # ----------------------------------------------------
        # RESET STATE
        # ----------------------------------------------------

        self._positive_count = 0

        self._latest_score = 0.0

        self._pcm_buffer.clear()

        self._wake_audio_buffer = np.zeros(
            0,
            dtype=np.float32,
        )

        self._reset_audio_filter()

        self._reset_wake_model()

        print(
            "[ZOE WAKE] "
            "Wakeword listener stopped."
        )

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_esp32_connected(self) -> bool:
        """
        Delegate connection state to the shared bridge if the
        bridge exposes it.
        """

        try:
            return bool(
                self._bridge.is_connected
            )

        except AttributeError:
            return False

    @property
    def latest_score(self) -> float:
        return self._latest_score


# ============================================================
# SIMPLE CLI TEST
# ============================================================

def main():

    def on_wake():

        print("")

        print(
            "========================================"
        )

        print(
            "       HEY ZOE DETECTED"
        )

        print(
            "========================================"
        )

        print("")

    wakeword = ZoeWakeWord(
        on_wake=on_wake,
        threshold=WAKEWORD_THRESHOLD,
    )

    try:

        wakeword.start()

        print(
            "[ZOE WAKE] "
            "Press Ctrl+C to stop."
        )

        while True:

            time.sleep(1.0)

    except KeyboardInterrupt:

        print(
            "\n[ZOE WAKE] "
            "Interrupted."
        )

    finally:

        wakeword.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()