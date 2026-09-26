from __future__ import annotations

import io
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfilt

from backend.listening import get_bridge

from dotenv import load_dotenv
from groq import Groq
from silero_vad import load_silero_vad
from speechbrain.inference.speaker import EncoderClassifier


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GROQ
# ============================================================

GROQ_MODEL = "whisper-large-v3-turbo"

GROQ_LANGUAGE = "en"

GROQ_TEMPERATURE = 0.0

GROQ_API_KEY1 = os.getenv("GROQ_API_KEY2")

GROQ_API_KEY2 = os.getenv("GROQ_API_KEY3")

GROQ_KEYS = [
    key
    for key in (
        GROQ_API_KEY1,
        GROQ_API_KEY2,
    )
    if key
]


_groq_key_index = 0

_groq_lock = threading.RLock()

_groq_clients: dict[str, Groq] = {}


# ============================================================
# GROQ CLIENT
# ============================================================

def _get_groq_client(
    api_key: str,
) -> Groq:

    with _groq_lock:

        client = _groq_clients.get(api_key)

        if client is None:

            client = Groq(
                api_key=api_key
            )

            _groq_clients[api_key] = client

        return client


if not GROQ_KEYS:

    print(
        "[ZOE GROQ] WARNING: "
        "No Groq API keys configured."
    )

else:

    print(
        "[ZOE GROQ] "
        f"{len(GROQ_KEYS)} API key(s) configured."
    )

    print(
        "[ZOE GROQ] "
        f"Model: {GROQ_MODEL}"
    )


# ============================================================
# AUDIO
# ============================================================

SAMPLE_RATE = 16000

CHANNELS = 1

BLOCK_SIZE = 512


# ============================================================
# AUDIO CLEANUP / NOISE REDUCTION
#
# Raw ESP32 PCM16
#       │
#       ▼
# 80 Hz high-pass
#       │
#       ▼
# Noise gate
#       │
#       ▼
# Automatic gain
#       │
#       ▼
# Soft limiter
#       │
#       ▼
# ZOE STT pipeline
#
# This is the same DSP processing used by the
# ESP32 recording pipeline.
# ============================================================

AUDIO_HIGHPASS_HZ = 80

AUDIO_GATE_THRESHOLD = 0.008

AUDIO_GATE_MIN_ATTENUATION = 0.05

AUDIO_TARGET_RMS = 0.08

AUDIO_MIN_RMS_FOR_AGC = 0.0005

AUDIO_MIN_GAIN = 0.7

AUDIO_MAX_GAIN = 3.0

AUDIO_LIMITER_DRIVE = 1.2


# ============================================================
# GROQ REALTIME
#
# Groq Whisper is HTTP based rather than token-streaming.
#
# Therefore ZOE periodically transcribes a rolling audio
# window while speech is active.
# ============================================================

REALTIME_INTERVAL_SECONDS = 0.85

REALTIME_WINDOW_SECONDS = 3.5

REALTIME_MIN_AUDIO_SECONDS = 0.75

REALTIME_MAX_AUDIO_SECONDS = 6.0

REALTIME_OVERLAP_SECONDS = 0.35


# ============================================================
# SILERO VAD
# ============================================================

VAD_SPEECH_THRESHOLD = 0.55

MIN_SPEECH_SECONDS = 0.30

END_SILENCE_SECONDS = 0.52

PRE_ROLL_SECONDS = 0.20

MAX_UTTERANCE_SECONDS = 30.0


# ============================================================
# SPEAKER VERIFICATION
# ============================================================

SPEAKER_MODEL = (
    "speechbrain/spkrec-ecapa-voxceleb"
)

VOICEPRINT_FILE = Path(
    "backend/data/hasin_voiceprint.pt"
)

SPEAKER_THRESHOLD = 0.50


# ============================================================
# REALTIME ECAPA
# ============================================================

SPEAKER_WINDOW_SECONDS = 0.50

SPEAKER_CHECK_INTERVAL = 0.10

SPEAKER_MIN_SECONDS = 0.30

SPEAKER_REQUIRED_POSITIVES = 1


# ============================================================
# BARGE-IN
# ============================================================

BARGE_IN_COOLDOWN_SECONDS = 0.35


# ============================================================
# WHISPER FILTER
# ============================================================

MAX_CONSECUTIVE_WORD_REPEAT = 2


# ============================================================
# RESULT
# ============================================================

@dataclass
class STTResult:

    text: str

    duration: float

    interrupted_tts: bool = False


# ============================================================
# ZOE STT
# ============================================================

class ZoeSTT:

    def __init__(
        self,
        model: str = GROQ_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",

        on_final: Optional[
            Callable[[str], None]
        ] = None,

        on_interruption: Optional[
            Callable[[], None]
        ] = None,

        on_speech_start: Optional[
            Callable[[], None]
        ] = None,

        on_speech_end: Optional[
            Callable[[], None]
        ] = None,

        on_partial: Optional[
            Callable[[str], None]
        ] = None,
    ):

        # ----------------------------------------------------
        # GROQ
        # ----------------------------------------------------

        self.model_name = model

        # Compatibility only.
        #
        # There is NO local Whisper model.
        self.device = device
        self.compute_type = compute_type

        # ----------------------------------------------------
        # CALLBACKS
        # ----------------------------------------------------

        self.on_final = on_final

        self.on_interruption = on_interruption

        self.on_speech_start = on_speech_start

        self.on_speech_end = on_speech_end

        self.on_partial = on_partial

        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        self._running = False

        self._thread: Optional[
            threading.Thread
        ] = None

        self._audio_queue: queue.Queue[
            np.ndarray
        ] = queue.Queue(
            maxsize=100
        )

        self._runtime_lock = threading.RLock()

        # ----------------------------------------------------
        # ESP32 AUDIO BRIDGE
        # ----------------------------------------------------

        self._bridge = get_bridge()

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
        # MICROPHONE
        # ----------------------------------------------------

        self._muted = False

        self._mute_lock = threading.RLock()

        # ----------------------------------------------------
        # TTS
        # ----------------------------------------------------

        self._speaking = False

        self._speaking_lock = threading.RLock()

        self._interruption_sent = False

        self._last_tts_stop_time = 0.0

        # ----------------------------------------------------
        # VAD
        # ----------------------------------------------------

        self._vad_model = None

        self._vad_buffer = np.zeros(
            0,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # SPEAKER MODEL
        # ----------------------------------------------------

        self._speaker_model = None

        self._hasin_voiceprint = None

        self._load_speaker_verification()

        # ----------------------------------------------------
        # ECAPA WORKER
        # ----------------------------------------------------

        self._speaker_queue: queue.Queue[
            Optional[np.ndarray]
        ] = queue.Queue(
            maxsize=1
        )

        self._speaker_thread: Optional[
            threading.Thread
        ] = None

        self._speaker_worker_running = False

        self._speaker_lock = threading.RLock()

        self._speaker_buffer = np.zeros(
            0,
            dtype=np.float32,
        )

        self._last_speaker_check = 0.0

        self._speaker_positive_count = 0

        self._latest_speaker_score = 0.0

        # ----------------------------------------------------
        # SPEECH STATE
        # ----------------------------------------------------

        self._speech_lock = threading.RLock()

        self._in_speech = False

        self._speech_started_at = 0.0

        self._last_voice_at = 0.0

        self._speech_interrupted_tts = False

        # ----------------------------------------------------
        # UTTERANCE
        # ----------------------------------------------------

        self._utterance_chunks: list[
            np.ndarray
        ] = []

        self._utterance_samples = 0

        # ----------------------------------------------------
        # PRE-ROLL
        # ----------------------------------------------------

        self._pre_roll: list[
            np.ndarray
        ] = []

        # ----------------------------------------------------
        # SPEECH GENERATION
        #
        # Protects against old Groq requests returning after
        # the utterance has already ended.
        # ----------------------------------------------------

        self._speech_generation = 0

        # ----------------------------------------------------
        # GROQ REALTIME WORKER
        # ----------------------------------------------------

        self._realtime_thread: Optional[
            threading.Thread
        ] = None

        self._realtime_running = False

        self._realtime_lock = threading.RLock()

        self._realtime_wake = threading.Event()

        self._realtime_audio_lock = threading.RLock()

        self._realtime_audio = np.zeros(
            0,
            dtype=np.float32,
        )

        self._latest_partial_text = ""

        self._last_realtime_request = 0.0

        # Prevent multiple Groq requests from running at
        # the same time.
        self._groq_realtime_lock = threading.Lock()

        # ----------------------------------------------------
        # INITIALIZE
        # ----------------------------------------------------

        self._load_vad()


    # ========================================================
    # AUDIO FILTER RESET
    # ========================================================

    def _reset_audio_filter(
        self,
    ):

        with self._audio_filter_lock:

            self._audio_filter_state = np.zeros(
                (
                    self._audio_filter.shape[0],
                    2,
                ),
                dtype=np.float64,
            )


    # ========================================================
    # ESP32 AUDIO PROCESSING
    #
    # Raw ESP32 PCM16 -> float32 -> DSP
    # ========================================================

    def _process_input_audio(
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
        # RMS
        # ----------------------------------------------------

        rms = np.sqrt(
            np.mean(
                audio * audio
            )
            + 1e-12
        )

        # ----------------------------------------------------
        # NOISE GATE
        # ----------------------------------------------------

        if (
            rms
            < AUDIO_GATE_THRESHOLD
        ):

            attenuation = max(
                AUDIO_GATE_MIN_ATTENUATION,
                rms
                / AUDIO_GATE_THRESHOLD,
            )

            audio *= attenuation

        # ----------------------------------------------------
        # AUTOMATIC GAIN
        # ----------------------------------------------------

        if (
            rms
            > AUDIO_MIN_RMS_FOR_AGC
        ):

            gain = (
                AUDIO_TARGET_RMS
                / rms
            )

            gain = np.clip(
                gain,
                AUDIO_MIN_GAIN,
                AUDIO_MAX_GAIN,
            )

            audio *= gain

        # ----------------------------------------------------
        # SOFT LIMITER
        # ----------------------------------------------------

        audio = np.tanh(
            audio
            * AUDIO_LIMITER_DRIVE
        )

        audio = np.clip(
            audio,
            -1.0,
            1.0,
        )

        return audio.astype(
            np.float32
        )


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

        sample_count = (
            len(data) // 2
        )

        if sample_count <= 0:

            return np.zeros(
                0,
                dtype=np.float32,
            )

        # Only use complete int16 samples.
        data = data[
            :sample_count * 2
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
    # ESP32 AUDIO CALLBACK
    # ========================================================

    def _on_audio_bytes(
        self,
        data: bytes,
    ):
        audio = self._pcm16_to_float32(data)

        if audio.size == 0:
            return

        audio = self._process_input_audio(audio)
        self._process_audio(audio)


    # ========================================================
    # GROQ AUDIO
    # ========================================================

    @staticmethod
    def _audio_to_wav_bytes(
        audio: np.ndarray,
    ) -> bytes:

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        audio = np.clip(
            audio,
            -1.0,
            1.0,
        )

        pcm = (
            audio * 32767.0
        ).astype(
            np.int16
        )

        buffer = io.BytesIO()

        wav.write(
            buffer,
            SAMPLE_RATE,
            pcm,
        )

        buffer.seek(0)

        return buffer.read()


    # ========================================================
    # GROQ TRANSCRIPTION
    # ========================================================

    def _transcribe_with_groq(
        self,
        audio: np.ndarray,
    ) -> Optional[str]:

        global _groq_key_index

        if not GROQ_KEYS:

            print(
                "[ZOE GROQ] "
                "No API keys available."
            )

            return None

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:

            return None

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        if duration < 0.05:

            return None

        audio_bytes = (
            self._audio_to_wav_bytes(
                audio
            )
        )

        with _groq_lock:

            starting_index = (
                _groq_key_index
                % len(GROQ_KEYS)
            )

        for attempt in range(
            len(GROQ_KEYS)
        ):

            index = (
                starting_index
                + attempt
            ) % len(GROQ_KEYS)

            api_key = GROQ_KEYS[index]

            key_number = index + 1

            try:

                client = (
                    _get_groq_client(
                        api_key
                    )
                )

                response = (
                    client.audio.transcriptions.create(

                        file=(
                            "zoe_audio.wav",
                            audio_bytes,
                        ),

                        model=self.model_name,

                        language=GROQ_LANGUAGE,

                        temperature=GROQ_TEMPERATURE,

                        response_format="json",

                    )
                )

                text = getattr(
                    response,
                    "text",
                    "",
                )

                text = self._clean_text(
                    text
                )

                if not text:

                    return None

                with _groq_lock:

                    _groq_key_index = index

                return text

            except Exception as e:

                print(
                    "[ZOE GROQ] "
                    f"Key {key_number} failed: "
                    f"{e}"
                )

                if (
                    attempt + 1
                    < len(GROQ_KEYS)
                ):

                    next_index = (
                        index + 1
                    ) % len(GROQ_KEYS)

                    print(
                        "[ZOE GROQ] "
                        f"Trying key "
                        f"{next_index + 1}..."
                    )

                    continue

        print(
            "[ZOE GROQ] "
            "All configured keys failed."
        )

        return None


    # ========================================================
    # REALTIME GROQ TRANSCRIPTION
    # ========================================================

    def _realtime_worker(
        self,
    ):

        print(
            "[ZOE GROQ] "
            "Realtime transcription worker started."
        )

        while True:

            with self._realtime_lock:

                if not self._realtime_running:

                    break

            self._realtime_wake.wait(
                timeout=0.10
            )

            self._realtime_wake.clear()

            with self._speech_lock:

                if not self._in_speech:

                    continue

                generation = (
                    self._speech_generation
                )

            with self._realtime_audio_lock:

                audio = (
                    self._realtime_audio.copy()
                )

            if audio.size == 0:

                continue

            duration = (
                len(audio)
                / SAMPLE_RATE
            )

            if (
                duration
                < REALTIME_MIN_AUDIO_SECONDS
            ):

                continue

            now = time.monotonic()

            if (
                now
                - self._last_realtime_request
                < REALTIME_INTERVAL_SECONDS
            ):

                continue

            if not self._groq_realtime_lock.acquire(
                blocking=False
            ):

                continue

            self._last_realtime_request = now

            try:

                with self._speech_lock:

                    if (
                        not self._in_speech
                        or generation
                        != self._speech_generation
                    ):

                        continue

                text = (
                    self._transcribe_with_groq(
                        audio
                    )
                )

                if not text:

                    continue

                text = self._clean_text(
                    text
                )

                if (
                    self._looks_like_hallucination(
                        text,
                        duration,
                    )
                ):

                    continue

                with self._speech_lock:

                    if (
                        not self._in_speech
                        or generation
                        != self._speech_generation
                    ):

                        continue

                with self._realtime_audio_lock:

                    current_audio_length = (
                        len(
                            self._realtime_audio
                        )
                    )

                with self._realtime_lock:

                    if (
                        text
                        == self._latest_partial_text
                    ):

                        continue

                    self._latest_partial_text = text

                print(
                    f"\r[ZOE PARTIAL] "
                    f"{text:<100}",
                    end="",
                    flush=True,
                )

                if self.on_partial:

                    try:

                        self.on_partial(
                            text
                        )

                    except Exception as e:

                        print(
                            "\n[ZOE STT] "
                            "Partial callback error: "
                            f"{e}"
                        )

                _ = current_audio_length

            finally:

                self._groq_realtime_lock.release()

        print(
            "[ZOE GROQ] "
            "Realtime transcription worker stopped."
        )


    # ========================================================
    # START REALTIME GROQ
    # ========================================================

    def _start_realtime_worker(
        self,
    ):

        with self._realtime_lock:

            if (
                self._realtime_thread
                is not None
                and self._realtime_thread.is_alive()
            ):

                self._realtime_running = True

                return True

            self._realtime_running = True

            thread = threading.Thread(
                target=self._realtime_worker,
                name="ZOE-Groq-Realtime",
                daemon=True,
            )

            self._realtime_thread = thread

        thread.start()

        return True


    # ========================================================
    # STOP REALTIME GROQ
    # ========================================================

    def _stop_realtime_worker(
        self,
    ):

        with self._realtime_lock:

            self._realtime_running = False

            self._realtime_wake.set()

            thread = self._realtime_thread

        if (
            thread is not None
            and thread
            is not threading.current_thread()
        ):

            thread.join(
                timeout=5.0
            )

            if thread.is_alive():

                print(
                    "[ZOE GROQ] "
                    "WARNING: realtime worker "
                    "did not stop within 5 seconds."
                )

        with self._realtime_lock:

            self._realtime_thread = None


    # ========================================================
    # FEED REALTIME AUDIO
    # ========================================================

    def _feed_realtime_audio(
        self,
        chunk: np.ndarray,
    ):

        with self._speech_lock:

            if not self._in_speech:

                return

        chunk = np.asarray(
            chunk,
            dtype=np.float32,
        )

        if chunk.size == 0:

            return

        maximum_samples = int(
            REALTIME_WINDOW_SECONDS
            * SAMPLE_RATE
        )

        with self._realtime_audio_lock:

            self._realtime_audio = (
                np.concatenate(
                    (
                        self._realtime_audio,
                        chunk,
                    )
                )
            )

            if (
                len(self._realtime_audio)
                > maximum_samples
            ):

                self._realtime_audio = (
                    self._realtime_audio[
                        -maximum_samples:
                    ]
                )

        self._realtime_wake.set()


    # ========================================================
    # RESET REALTIME
    # ========================================================

    def _reset_realtime(
        self,
    ):

        with self._realtime_audio_lock:

            self._realtime_audio = np.zeros(
                0,
                dtype=np.float32,
            )

        with self._realtime_lock:

            self._latest_partial_text = ""

            self._last_realtime_request = 0.0

        if self.on_partial:

            try:

                self.on_partial("")

            except Exception as e:

                print(
                    "[ZOE STT] "
                    "Partial reset callback error: "
                    f"{e}"
                )


    # ========================================================
    # SPEAKER VERIFICATION
    # ========================================================

    def _load_speaker_verification(
        self,
    ):

        try:

            print(
                "[ZOE SPEAKER] "
                "Loading ECAPA-TDNN..."
            )

            if not VOICEPRINT_FILE.exists():

                raise FileNotFoundError(
                    "Hasin voiceprint not found: "
                    f"{VOICEPRINT_FILE}"
                )

            self._speaker_model = (
                EncoderClassifier.from_hparams(
                    source=SPEAKER_MODEL,
                    savedir="backend/models/ecapa",
                )
            )

            saved = torch.load(
                VOICEPRINT_FILE,
                map_location="cpu",
                weights_only=False,
            )

            if not isinstance(
                saved,
                dict,
            ):

                raise TypeError(
                    "Voiceprint file must contain "
                    "a dictionary."
                )

            if "embedding" not in saved:

                raise KeyError(
                    "Voiceprint does not contain "
                    "'embedding'."
                )

            voiceprint = saved[
                "embedding"
            ]

            voiceprint = (
                voiceprint
                .detach()
                .float()
            )

            voiceprint = F.normalize(
                voiceprint,
                dim=0,
            )

            self._hasin_voiceprint = (
                voiceprint
            )

            print(
                "[ZOE SPEAKER] "
                "ECAPA-TDNN ready."
            )

            print(
                "[ZOE SPEAKER] "
                f"Threshold: "
                f"{SPEAKER_THRESHOLD:.2f}"
            )

        except Exception as e:

            print(
                "[ZOE SPEAKER] "
                "Failed to load:"
            )

            print(
                f"[ZOE SPEAKER] {e}"
            )

            self._speaker_model = None

            self._hasin_voiceprint = None


    # ========================================================
    # SPEAKER SCORE
    # ========================================================

    def _speaker_score(
        self,
        audio: np.ndarray,
    ) -> float:

        if (
            self._speaker_model is None
            or self._hasin_voiceprint is None
        ):

            return 0.0

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        if duration < SPEAKER_MIN_SECONDS:

            return 0.0

        try:

            audio = (
                audio
                - np.mean(audio)
            )

            peak = float(
                np.max(
                    np.abs(audio)
                )
            )

            if peak > 1.0:

                audio = (
                    audio
                    / peak
                )

            audio_tensor = (
                torch.from_numpy(
                    audio
                )
                .unsqueeze(0)
            )

            with torch.no_grad():

                embedding = (
                    self._speaker_model
                    .encode_batch(
                        audio_tensor
                    )
                )

            embedding = (
                embedding
                .squeeze()
                .float()
            )

            embedding = F.normalize(
                embedding,
                dim=0,
            )

            return float(
                torch.dot(
                    self._hasin_voiceprint,
                    embedding,
                ).item()
            )

        except Exception as e:

            print(
                "[ZOE SPEAKER] "
                f"Score error: {e}"
            )

            return 0.0


    # ========================================================
    # COMPLETE SPEAKER VERIFICATION
    # ========================================================

    def _verify_speaker(
        self,
        audio: np.ndarray,
    ) -> tuple[bool, float]:

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        if duration < SPEAKER_MIN_SECONDS:

            print(
                "[ZOE SPEAKER] "
                f"Audio too short "
                f"({duration:.2f}s). Rejected."
            )

            return False, 0.0

        score = self._speaker_score(
            audio
        )

        is_hasin = (
            score
            >= SPEAKER_THRESHOLD
        )

        if is_hasin:

            print(
                "[ZOE SPEAKER] "
                f"HASIN ✓ score={score:.4f}"
            )

        else:

            print(
                "[ZOE SPEAKER] "
                f"OTHER SPEAKER ✗ "
                f"score={score:.4f}"
            )

        return is_hasin, score


    # ========================================================
    # RESET SPEAKER
    # ========================================================

    def _reset_speaker_detection(
        self,
    ):

        with self._speaker_lock:

            self._speaker_buffer = np.zeros(
                0,
                dtype=np.float32,
            )

            self._last_speaker_check = 0.0

            self._speaker_positive_count = 0

            self._latest_speaker_score = 0.0

        while True:

            try:

                self._speaker_queue.get_nowait()

            except queue.Empty:

                break


    # ========================================================
    # FEED ECAPA
    # ========================================================

    def _feed_speaker_detector(
        self,
        chunk: np.ndarray,
    ):

        if (
            self._speaker_model is None
            or self._hasin_voiceprint is None
        ):

            return

        with self._speaking_lock:

            if not self._speaking:

                return

        with self._speaker_lock:

            self._speaker_buffer = (
                np.concatenate(
                    (
                        self._speaker_buffer,
                        chunk,
                    )
                )
            )

            required_samples = int(
                SPEAKER_WINDOW_SECONDS
                * SAMPLE_RATE
            )

            if (
                len(self._speaker_buffer)
                < required_samples
            ):

                return

            now = time.monotonic()

            if (
                now
                - self._last_speaker_check
                < SPEAKER_CHECK_INTERVAL
            ):

                return

            self._last_speaker_check = now

            window = (
                self._speaker_buffer[
                    -required_samples:
                ].copy()
            )

        try:

            self._speaker_queue.put_nowait(
                window
            )

        except queue.Full:

            try:

                self._speaker_queue.get_nowait()

            except queue.Empty:

                pass

            try:

                self._speaker_queue.put_nowait(
                    window
                )

            except queue.Full:

                pass


    # ========================================================
    # ECAPA WORKER
    # ========================================================

    def _speaker_worker(
        self,
    ):

        print(
            "[ZOE SPEAKER] "
            "Realtime speaker worker started."
        )

        while self._speaker_worker_running:

            try:

                audio = (
                    self._speaker_queue.get(
                        timeout=0.1
                    )
                )

            except queue.Empty:

                continue

            if audio is None:

                break

            with self._speaking_lock:

                if not self._speaking:

                    continue

            score = self._speaker_score(
                audio
            )

            with self._speaker_lock:

                self._latest_speaker_score = score

            if score >= SPEAKER_THRESHOLD:

                with self._speaker_lock:

                    self._speaker_positive_count += 1

                    positives = (
                        self._speaker_positive_count
                    )

                if (
                    positives
                    >= SPEAKER_REQUIRED_POSITIVES
                ):

                    with self._speaker_lock:

                        self._speaker_positive_count = 0

                    self._trigger_hasin_barge_in()

            else:

                with self._speaker_lock:

                    self._speaker_positive_count = 0

        print(
            "[ZOE SPEAKER] "
            "Realtime speaker worker stopped."
        )


    # ========================================================
    # BARGE-IN
    # ========================================================

    def _trigger_hasin_barge_in(
        self,
    ):

        now = time.monotonic()

        with self._speaking_lock:

            if not self._speaking:

                return

            if self._interruption_sent:

                return

            if (
                now
                - self._last_tts_stop_time
                < BARGE_IN_COOLDOWN_SECONDS
            ):

                return

            self._interruption_sent = True

            self._speaking = False

            self._last_tts_stop_time = now

        with self._speaker_lock:

            score = (
                self._latest_speaker_score
            )

        with self._speech_lock:

            self._speech_interrupted_tts = True

        print()
        print(
            "========================================"
        )
        print(
            "[ZOE BARGE] HASIN DETECTED"
        )
        print(
            f"[ZOE BARGE] ECAPA score={score:.4f}"
        )
        print(
            "[ZOE BARGE] STOPPING ZOE NOW"
        )
        print(
            "[ZOE BARGE] RECORDING CONTINUES"
        )
        print(
            "========================================"
        )
        print()

        if self.on_interruption:

            try:

                self.on_interruption()

            except Exception as e:

                print(
                    "[ZOE STT] "
                    f"Interruption callback error: {e}"
                )


    # ========================================================
    # VAD
    # ========================================================

    def _load_vad(
        self,
    ):

        try:

            print(
                "[ZOE VAD] "
                "Loading Silero VAD..."
            )

            self._vad_model = (
                load_silero_vad()
            )

            print(
                "[ZOE VAD] "
                "Silero VAD ready."
            )

        except Exception as e:

            print(
                "[ZOE VAD] "
                f"Failed to load: {e}"
            )

            self._vad_model = None


    # ========================================================
    # START
    # ========================================================

    def start(
        self,
    ):

        with self._runtime_lock:

            if self._running:

                return

            self._running = True

            with self._speech_lock:

                self._speech_generation += 1

                self._in_speech = False

                self._speech_interrupted_tts = False

            self._reset_utterance()

            self._reset_realtime()

            self._vad_buffer = np.zeros(
                0,
                dtype=np.float32,
            )

            self._pre_roll.clear()

            self._reset_speaker_detection()

            self._reset_audio_filter()

            self._clear_audio_queue()

            # ------------------------------------------------
            # ESP32 AUDIO BRIDGE
            # ------------------------------------------------

            self._bridge.subscribe(
                self._on_audio_bytes
            )

            # ------------------------------------------------
            # ECAPA
            # ------------------------------------------------

            self._speaker_worker_running = True

            self._speaker_thread = (
                threading.Thread(
                    target=self._speaker_worker,
                    name="ZOE-Speaker",
                    daemon=True,
                )
            )

            self._speaker_thread.start()

            # ------------------------------------------------
            # GROQ REALTIME
            # ------------------------------------------------

            self._start_realtime_worker()

            # ------------------------------------------------
            # AUDIO WORKER
            # ------------------------------------------------

            self._thread = (
                threading.Thread(
                    target=self._run,
                    name="ZOE-STT",
                    daemon=True,
                )
            )

            self._thread.start()

        print(
            "[ZOE STT] Listening."
        )

        print(
            "[ZOE STT] "
            "ESP32 INMP441 microphone active."
        )

        print(
            "[ZOE STT] "
            "Shared ESP32 audio bridge active."
        )

        print(
            "[ZOE STT] "
            f"Groq Whisper: {GROQ_MODEL}"
        )

        print(
            "[ZOE STT] "
            "No local Whisper model."
        )

        print(
            "[ZOE STT] "
            "Groq handles realtime + final transcription."
        )


    # ========================================================
    # STOP
    # ========================================================

    def stop(
        self,
    ):

        with self._runtime_lock:

            if not self._running:

                return

            print(
                "[ZOE STT] Stopping..."
            )

            self._running = False

            with self._speech_lock:

                was_in_speech = (
                    self._in_speech
                )

                self._in_speech = False

                self._speech_generation += 1

                self._speech_interrupted_tts = False

            if (
                was_in_speech
                and self.on_speech_end
            ):

                try:

                    self.on_speech_end()

                except Exception as e:

                    print(
                        "[ZOE STT] "
                        f"Speech-end callback error: {e}"
                    )

            # ------------------------------------------------
            # ESP32 AUDIO BRIDGE
            # ------------------------------------------------

            self._bridge.unsubscribe(
                self._on_audio_bytes
            )

            # ------------------------------------------------
            # REALTIME GROQ
            # ------------------------------------------------

            self._stop_realtime_worker()

            # ------------------------------------------------
            # ECAPA
            # ------------------------------------------------

            self._speaker_worker_running = False

            try:

                self._speaker_queue.put_nowait(
                    None
                )

            except queue.Full:

                pass

            if self._speaker_thread is not None:

                if (
                    self._speaker_thread
                    is not threading.current_thread()
                ):

                    self._speaker_thread.join(
                        timeout=5.0
                    )

            self._speaker_thread = None

            # ------------------------------------------------
            # AUDIO WORKER
            # ------------------------------------------------

            if self._thread is not None:

                if (
                    self._thread
                    is not threading.current_thread()
                ):

                    self._thread.join(
                        timeout=5.0
                    )

                    if self._thread.is_alive():

                        print(
                            "[ZOE STT] "
                            "WARNING: audio worker "
                            "did not exit within 5 seconds."
                        )

            self._thread = None

            # ------------------------------------------------
            # RESET
            # ------------------------------------------------

            self._reset_utterance()

            self._pre_roll.clear()

            self._vad_buffer = np.zeros(
                0,
                dtype=np.float32,
            )

            self._reset_speaker_detection()

            self._reset_audio_filter()

            self._clear_audio_queue()

            self._reset_realtime()

        print(
            "[ZOE STT] Stopped."
        )


    # ========================================================
    # MUTE
    # ========================================================

    def set_muted(
        self,
        muted: bool,
    ):

        muted = bool(muted)

        with self._mute_lock:

            previous = self._muted

            if previous == muted:

                return

            self._muted = muted

        if muted:

            print(
                "[ZOE STT] "
                "Microphone muted."
            )

            with self._speech_lock:

                was_speaking = (
                    self._in_speech
                )

                self._in_speech = False

                self._speech_generation += 1

            self._reset_utterance()

            self._reset_realtime()

            self._pre_roll.clear()

            self._vad_buffer = np.zeros(
                0,
                dtype=np.float32,
            )

            self._reset_speaker_detection()

            self._reset_audio_filter()

            self._clear_audio_queue()

            if (
                was_speaking
                and self.on_speech_end
            ):

                try:

                    self.on_speech_end()

                except Exception as e:

                    print(
                        "[ZOE STT] "
                        f"Speech-end callback error: {e}"
                    )

        else:

            print(
                "[ZOE STT] "
                "Microphone unmuted."
            )

            with self._speech_lock:

                self._speech_generation += 1

                self._in_speech = False

            self._reset_utterance()

            self._reset_realtime()

            self._vad_buffer = np.zeros(
                0,
                dtype=np.float32,
            )

            self._pre_roll.clear()

            self._reset_speaker_detection()

            self._reset_audio_filter()

            self._clear_audio_queue()


    # ========================================================

    def is_muted(
        self,
    ) -> bool:

        with self._mute_lock:

            return self._muted


    # ========================================================
    # TTS STATE
    # ========================================================

    def set_speaking(
        self,
        speaking: bool,
    ):

        speaking = bool(speaking)

        with self._speaking_lock:

            previous = self._speaking

            self._speaking = speaking

            if (
                speaking
                and not previous
            ):

                self._interruption_sent = False

                self._reset_speaker_detection()

                print(
                    "[ZOE STT] "
                    "TTS started."
                )

                print(
                    "[ZOE STT] "
                    "Realtime Hasin "
                    "barge-in active."
                )

            elif (
                not speaking
                and previous
            ):

                self._last_tts_stop_time = (
                    time.monotonic()
                )

                self._reset_speaker_detection()

                print(
                    "[ZOE STT] "
                    "TTS ended."
                )


    # ========================================================

    def is_speaking(
        self,
    ) -> bool:

        with self._speaking_lock:

            return self._speaking


    # ========================================================

    def is_running(
        self,
    ) -> bool:

        with self._runtime_lock:

            return self._running


    # ========================================================
    # AUDIO WORKER
    #
    # ESP32 INMP441
    #       │
    #       ▼
    #      TCP
    #       │
    #       ▼
    #   PCM16 -> float32
    #       │
    #       ▼
    #   DSP / NOISE REDUCTION
    #       │
    #       ▼
    #   _process_audio()
    #       │
    #       ├── Silero VAD
    #       ├── ECAPA
    #       └── Groq Whisper
    # ========================================================

    def _run(
        self,
    ):

        while self._running:
            time.sleep(0.1)


    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    def _process_audio(
        self,
        chunk: np.ndarray,
    ):

        if chunk.size == 0:

            return

        if self.is_muted():

            return

        now = time.monotonic()

        # ----------------------------------------------------
        # GROQ REALTIME AUDIO
        #
        # Only retain audio while the user is actually
        # speaking.
        # ----------------------------------------------------

        with self._speech_lock:

            currently_speaking = (
                self._in_speech
            )

        if currently_speaking:

            self._feed_realtime_audio(
                chunk
            )

        # ----------------------------------------------------
        # VAD
        # ----------------------------------------------------

        is_speech = (
            self._detect_speech(
                chunk
            )
        )

        # ====================================================
        # SPEECH
        # ====================================================

        if is_speech:

            if not self._in_speech:

                self._start_speech()

            self._last_voice_at = now

            self._utterance_chunks.append(
                chunk
            )

            self._utterance_samples += (
                len(chunk)
            )

            self._feed_speaker_detector(
                chunk
            )

            duration = (
                self._utterance_samples
                / SAMPLE_RATE
            )

            if (
                duration
                >= MAX_UTTERANCE_SECONDS
            ):

                print(
                    "[ZOE VAD] "
                    "Maximum utterance reached."
                )

                self._finish_speech()

            self._update_pre_roll(
                chunk
            )

            return

        # ====================================================
        # SILENCE
        # ====================================================

        if self._in_speech:

            self._utterance_chunks.append(
                chunk
            )

            self._utterance_samples += (
                len(chunk)
            )

            silence_duration = (
                now
                - self._last_voice_at
            )

            if (
                silence_duration
                >= END_SILENCE_SECONDS
            ):

                self._finish_speech()

            self._update_pre_roll(
                chunk
            )


    # ========================================================
    # PRE-ROLL
    # ========================================================

    def _update_pre_roll(
        self,
        chunk: np.ndarray,
    ):

        self._pre_roll.append(
            chunk
        )

        maximum = int(
            PRE_ROLL_SECONDS
            * SAMPLE_RATE
        )

        total = sum(
            len(x)
            for x in self._pre_roll
        )

        while total > maximum:

            removed = (
                self._pre_roll.pop(0)
            )

            total -= len(
                removed
            )


    # ========================================================
    # VAD
    # ========================================================

    def _detect_speech(
        self,
        chunk: np.ndarray,
    ) -> bool:

        if self._vad_model is None:

            return False

        self._vad_buffer = (
            np.concatenate(
                (
                    self._vad_buffer,
                    chunk,
                )
            )
        )

        window_size = 512

        if (
            len(self._vad_buffer)
            < window_size
        ):

            return False

        window = (
            self._vad_buffer[
                :window_size
            ]
        )

        self._vad_buffer = (
            self._vad_buffer[
                window_size:
            ]
        )

        try:

            audio_tensor = (
                torch.from_numpy(
                    window
                )
            )

            with torch.no_grad():

                probability = float(
                    self._vad_model(
                        audio_tensor,
                        SAMPLE_RATE,
                    ).item()
                )

            return (
                probability
                >= VAD_SPEECH_THRESHOLD
            )

        except Exception as e:

            print(
                "[ZOE VAD] "
                f"Inference error: {e}"
            )

            return False


    # ========================================================
    # START SPEECH
    # ========================================================

    def _start_speech(
        self,
    ):

        with self._speech_lock:

            if self._in_speech:

                return

            self._in_speech = True

            self._speech_generation += 1

            generation = (
                self._speech_generation
            )

            now = time.monotonic()

            self._speech_started_at = now

            self._last_voice_at = now

            self._speech_interrupted_tts = (
                self.is_speaking()
            )

        # ----------------------------------------------------
        # Build utterance from pre-roll.
        # ----------------------------------------------------

        self._utterance_chunks = (
            list(self._pre_roll)
        )

        self._utterance_samples = sum(
            len(x)
            for x in self._utterance_chunks
        )

        # ----------------------------------------------------
        # Reset realtime Groq buffer.
        # ----------------------------------------------------

        self._reset_realtime()

        print(
            "[ZOE VAD] "
            f"Speech started "
            f"(generation={generation})."
        )

        if self.on_speech_start:

            try:

                self.on_speech_start()

            except Exception as e:

                print(
                    "[ZOE STT] "
                    f"Speech-start callback error: {e}"
                )


    # ========================================================
    # FINISH SPEECH
    # ========================================================

    def _finish_speech(
        self,
    ):

        with self._speech_lock:

            if not self._in_speech:

                return

            self._in_speech = False

            self._speech_generation += 1

            generation = (
                self._speech_generation
            )

            interrupted_tts = (
                self._speech_interrupted_tts
            )

            self._speech_interrupted_tts = False

            chunks = list(
                self._utterance_chunks
            )

        # ----------------------------------------------------
        # Physical speech ended.
        # ----------------------------------------------------

        if self.on_speech_end:

            try:

                self.on_speech_end()

            except Exception as e:

                print(
                    "[ZOE STT] "
                    f"Speech-end callback error: {e}"
                )

        self._reset_utterance()

        if not chunks:

            self._reset_realtime()

            return

        audio = np.concatenate(
            chunks
        ).astype(
            np.float32
        )

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        print()
        print(
            "----------------------------------------"
        )
        print(
            "[ZOE STT] Utterance complete"
        )
        print(
            f"[ZOE STT] Duration: {duration:.2f}s"
        )
        print(
            f"[ZOE STT] Final: Groq {self.model_name}"
        )
        print(
            "----------------------------------------"
        )

        if duration < MIN_SPEECH_SECONDS:

            print(
                "[ZOE STT] "
                "Utterance too short."
            )

            self._reset_realtime()

            return

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # There is NO finalization queue.
        # There is NO finalization worker.
        #
        # The audio worker directly performs the same Groq
        # transcription used by realtime STT.
        # ----------------------------------------------------

        self._transcribe_final(
            audio,
            duration,
            interrupted_tts,
            generation,
        )


    # ========================================================
    # FINAL GROQ TRANSCRIPTION
    #
    # Same engine as realtime.
    # ========================================================

    def _transcribe_final(
        self,
        audio: np.ndarray,
        duration: float,
        interrupted_tts: bool,
        generation: int,
    ):

        # ----------------------------------------------------
        # Verify speaker first.
        # ----------------------------------------------------

        print(
            "[ZOE SPEAKER] "
            "Verifying complete utterance..."
        )

        is_hasin, score = (
            self._verify_speaker(
                audio
            )
        )

        with self._speech_lock:

            if (
                generation
                != self._speech_generation
            ):

                return

        if not is_hasin:

            print(
                "[ZOE STT] "
                "Speaker rejected."
            )

            self._reset_realtime()

            return

        print(
            "[ZOE SPEAKER] "
            "Speaker verified."
        )

        # ----------------------------------------------------
        # SAME GROQ MODEL
        # ----------------------------------------------------

        print(
            "[ZOE STT] "
            f"Groq Whisper "
            f"{self.model_name} transcription..."
        )

        text = (
            self._transcribe_with_groq(
                audio
            )
        )

        if not text:

            print(
                "[ZOE STT] "
                "Groq returned no words."
            )

            self._reset_realtime()

            return

        text = (
            self._remove_repeated_text(
                text
            )
        )

        text = self._clean_text(
            text
        )

        if not text:

            self._reset_realtime()

            return

        if self._looks_like_hallucination(
            text,
            duration,
        ):

            print(
                "[ZOE STT] "
                f"Ignored hallucination: "
                f"{text!r}"
            )

            self._reset_realtime()

            return

        # ----------------------------------------------------
        # Generation guard.
        # ----------------------------------------------------

        with self._speech_lock:

            if (
                generation
                != self._speech_generation
            ):

                return

        result = STTResult(

            text=text,

            duration=duration,

            interrupted_tts=(
                interrupted_tts
            ),

        )

        print(
            "[ZOE STT] "
            f"Final: {result.text}"
        )

        if result.interrupted_tts:

            print(
                "[ZOE STT] "
                "✓ Barge-in command accepted."
            )

        if self.on_final:

            try:

                self.on_final(
                    result.text
                )

            except Exception as e:

                print(
                    "[ZOE STT] "
                    f"Final callback error: {e}"
                )

        self._reset_realtime()


    # ========================================================
    # RESET UTTERANCE
    # ========================================================

    def _reset_utterance(
        self,
    ):

        self._utterance_chunks = []

        self._utterance_samples = 0


    # ========================================================
    # REMOVE REPEATED TEXT
    # ========================================================

    @staticmethod
    def _remove_repeated_text(
        text: str,
        max_repeat=MAX_CONSECUTIVE_WORD_REPEAT,
    ) -> str:

        output = []

        for token in text.split():

            normalized = (
                token
                .strip(
                    ".,!?;:"
                )
                .lower()
            )

            if (
                normalized
                and len(output)
                >= max_repeat
                and all(
                    previous
                    .strip(
                        ".,!?;:"
                    )
                    .lower()
                    == normalized
                    for previous
                    in output[
                        -max_repeat:
                    ]
                )
            ):

                continue

            output.append(token)

        return " ".join(
            output
        )


    # ========================================================
    # CLEAN TEXT
    # ========================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        return " ".join(
            str(text).split()
        ).strip()


    # ========================================================
    # HALLUCINATION FILTER
    # ========================================================

    @staticmethod
    def _looks_like_hallucination(
        text: str,
        duration: float,
    ) -> bool:

        words = (
            text
            .lower()
            .split()
        )

        if not words:

            return True

        if (
            duration < 0.7
            and len(words) > 8
        ):

            return True

        if (
            len(words) >= 3
            and len(set(words)) == 1
        ):

            return True

        hallucinations = {

            "thank you",

            "thanks",

            "thanks for watching",

            "thank you for watching",

            "you",

            "bye",

        }

        if (
            duration < 1.0
            and text.lower()
            in hallucinations
        ):

            return True

        return False


    # ========================================================
    # CLEAR AUDIO QUEUE
    # ========================================================

    def _clear_audio_queue(
        self,
    ):

        while True:

            try:

                self._audio_queue.get_nowait()

            except queue.Empty:

                break


    # ========================================================
    # PUBLIC SPEAKER SCORE
    # ========================================================

    def get_speaker_score(
        self,
    ) -> float:

        with self._speaker_lock:

            return (
                self._latest_speaker_score
            )


    # ========================================================
    # PUBLIC PARTIAL TEXT
    # ========================================================

    def get_partial_text(
        self,
    ) -> str:

        with self._realtime_lock:

            return (
                self._latest_partial_text
            )


    # ========================================================
    # PUBLIC SPEECH STATE
    # ========================================================

    def is_user_speaking(
        self,
    ) -> bool:

        with self._speech_lock:

            return self._in_speech


    # ========================================================
    # ESP32 CONNECTION STATE
    # ========================================================

    def is_esp32_connected(
        self,
    ) -> bool:

        connected = getattr(
            self._bridge,
            "is_connected",
            False,
        )

        if callable(connected):
            try:
                return bool(connected())
            except Exception:
                return False

        return bool(connected)


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_stt: Optional[
    ZoeSTT
] = None

_stt_global_lock = threading.RLock()


# ============================================================
# START
# ============================================================

def start(
    on_final: Optional[
        Callable[[str], None]
    ] = None,

    on_interruption: Optional[
        Callable[[], None]
    ] = None,

    on_speech_start: Optional[
        Callable[[], None]
    ] = None,

    on_speech_end: Optional[
        Callable[[], None]
    ] = None,

    on_partial: Optional[
        Callable[[str], None]
    ] = None,
):

    global _stt

    with _stt_global_lock:

        if _stt is None:

            _stt = ZoeSTT(

                model=GROQ_MODEL,

                device="cpu",

                compute_type="int8",

                on_final=on_final,

                on_interruption=(
                    on_interruption
                ),

                on_speech_start=(
                    on_speech_start
                ),

                on_speech_end=(
                    on_speech_end
                ),

                on_partial=(
                    on_partial
                ),
            )

        else:

            if on_final is not None:

                _stt.on_final = on_final

            if on_interruption is not None:

                _stt.on_interruption = (
                    on_interruption
                )

            if on_speech_start is not None:

                _stt.on_speech_start = (
                    on_speech_start
                )

            if on_speech_end is not None:

                _stt.on_speech_end = (
                    on_speech_end
                )

            if on_partial is not None:

                _stt.on_partial = on_partial

        _stt.start()

        return _stt


# ============================================================
# STOP
# ============================================================

def stop():

    with _stt_global_lock:

        if _stt is not None:

            _stt.stop()


# ============================================================
# TTS STATE
# ============================================================

def set_speaking(
    speaking: bool,
):

    with _stt_global_lock:

        if _stt is not None:

            _stt.set_speaking(
                speaking
            )


def is_speaking() -> bool:

    with _stt_global_lock:

        if _stt is None:

            return False

        return _stt.is_speaking()


# ============================================================
# MUTE
# ============================================================

def set_muted(
    muted: bool,
):

    with _stt_global_lock:

        if _stt is not None:

            _stt.set_muted(
                muted
            )


def is_muted() -> bool:

    with _stt_global_lock:

        if _stt is None:

            return False

        return _stt.is_muted()


# ============================================================
# RUNNING
# ============================================================

def is_running() -> bool:

    with _stt_global_lock:

        if _stt is None:

            return False

        return _stt.is_running()


# ============================================================
# PARTIAL
# ============================================================

def get_partial_text() -> str:

    with _stt_global_lock:

        if _stt is None:

            return ""

        return _stt.get_partial_text()


# ============================================================
# USER SPEECH
# ============================================================

def is_user_speaking() -> bool:

    with _stt_global_lock:

        if _stt is None:

            return False

        return _stt.is_user_speaking()


# ============================================================
# SPEAKER SCORE
# ============================================================

def get_speaker_score() -> float:

    with _stt_global_lock:

        if _stt is None:

            return 0.0

        return _stt.get_speaker_score()


# ============================================================
# ESP32 CONNECTION
# ============================================================

def is_esp32_connected() -> bool:

    with _stt_global_lock:

        if _stt is None:

            return False

        return _stt.is_esp32_connected()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    def final_text(
        text: str,
    ):

        print()
        print()
        print(
            "╔══════════════════════════════════════╗"
        )
        print(
            "║           FINAL TRANSCRIPT           ║"
        )
        print(
            "╚══════════════════════════════════════╝"
        )
        print()
        print(
            f"  {text}"
        )
        print()


    def partial_text(
        text: str,
    ):

        if text:

            print(
                f"\rLIVE: "
                f"{text:<100}",
                end="",
                flush=True,
            )

        else:

            print(
                "\r"
                + " " * 120
                + "\r",
                end="",
                flush=True,
            )


    def interrupted():

        print()
        print(
            "╔══════════════════════════════════════╗"
        )
        print(
            "║        USER INTERRUPTED ZOE          ║"
        )
        print(
            "╚══════════════════════════════════════╝"
        )
        print()


    def speech_started():

        print()
        print(
            "╔══════════════════════════════════════╗"
        )
        print(
            "║          USER IS SPEAKING            ║"
        )
        print(
            "╚══════════════════════════════════════╝"
        )
        print()


    def speech_ended():

        print()
        print(
            "[ZOE] Physical speech ended."
        )


    engine = start(

        on_final=final_text,

        on_interruption=interrupted,

        on_speech_start=speech_started,

        on_speech_end=speech_ended,

        on_partial=partial_text,

    )

    print()
    print(
        "=" * 60
    )
    print(
        "ZOE — ESP32 GROQ-ONLY SPEECH-TO-TEXT"
    )
    print(
        "=" * 60
    )
    print()

    print(
        f"STT           : Groq {GROQ_MODEL}"
    )

    print(
        "Realtime      : Groq rolling transcription"
    )

    print(
        "VAD           : Silero"
    )

    print(
        "Speaker       : ECAPA"
    )

    print(
        "Microphone    : ESP32 + INMP441"
    )

    print(
        "ESP32         : Shared audio bridge"
    )

    print()

    print(
        "Audio architecture:"
    )

    print(
        "  ESP32 INMP441 microphone"
    )

    print(
        "          │"
    )

    print(
        "          ▼"
    )

    print(
        "       TCP PCM16"
    )

    print(
        "          │"
    )

    print(
        "          ▼"
    )

    print(
        "     ZOE DSP cleanup"
    )

    print(
        "          │"
    )

    print(
        "          ├── 80 Hz high-pass"
    )

    print(
        "          ├── Noise gate"
    )

    print(
        "          ├── Automatic gain"
    )

    print(
        "          └── Soft limiter"
    )

    print(
        "          │"
    )

    print(
        "          ├── Silero VAD"
    )

    print(
        "          ├── ECAPA barge-in"
    )

    print(
        "          └── Groq Whisper"
    )

    print(
        "                ├── realtime"
    )

    print(
        "                └── final"
    )

    print()

    print(
        "Lifecycle:"
    )

    print(
        "  ESP32 owns microphone"
    )

    print(
        "  ESP32 streams raw PCM16 over TCP"
    )

    print(
        "  ZOE performs audio cleanup"
    )

    print(
        "  Groq is the only STT engine"
    )

    print(
        "  No local Whisper"
    )

    print(
        "  No RealtimeSTT"
    )

    print(
        "  No finalization worker"
    )

    print(
        "  Mute = stop processing microphone"
    )

    print(
        "  Unmute = resume processing"
    )

    print()

    print(
        "Waiting for ESP32 microphone..."
    )

    print(
        "Speak naturally."
    )

    print(
        "Ctrl+C to stop."
    )

    print()

    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        print()

        print(
            "[ZOE] Shutting down..."
        )

        stop()