
#!/usr/bin/env python3
"""
Build ZOE's Hasin speaker voiceprint from recordings in voice_samples/.

Run from the ZOE repository root:
    python -m backend.voice_test

Output:
    backend/data/hasin_voiceprint.pt

Dependencies:
    python -m pip install torch torchaudio speechbrain numpy

External dependency:
    FFmpeg must be installed and available on PATH.

The output contains an L2-normalized ECAPA-TDNN embedding compatible
with a ZOE speaker-verification loader that expects an "embedding" tensor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16_000
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MIN_DURATION_SECONDS = 1.5
DECODE_TIMEOUT_SECONDS = 120

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
}

# This file is backend/voice_test.py.
PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLES_DIR = PROJECT_ROOT / "voice_samples"
OUTPUT_FILE = PROJECT_ROOT / "data" / "hasin_voiceprint.pt"
MODEL_CACHE = PROJECT_ROOT / "models" / "ecapa"


# ============================================================
# AUDIO DECODING
# ============================================================

def load_audio(path: Path) -> torch.Tensor:
    """
    Decode an audio file using FFmpeg.

    Returns:
        A mono, float32 PyTorch tensor sampled at 16 kHz.
    """

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=DECODE_TIMEOUT_SECONDS,
        )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and reopen PowerShell."
        ) from exc

    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(
            "utf-8", errors="replace"
        ).strip()

        raise RuntimeError(
            f"FFmpeg could not decode this file: {detail}"
        ) from exc

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Audio decoding exceeded {DECODE_TIMEOUT_SECONDS} seconds."
        ) from exc

    # FFmpeg outputs little-endian 32-bit floating-point PCM.
    audio_array = np.frombuffer(
        result.stdout,
        dtype="<f4",
    ).copy()

    if audio_array.size == 0:
        raise ValueError("Decoded audio is empty.")

    if not np.isfinite(audio_array).all():
        raise ValueError("Audio contains NaN or infinite samples.")

    waveform = torch.from_numpy(audio_array).to(
        dtype=torch.float32
    )

    # Remove DC offset, matching the basic runtime preprocessing.
    waveform = waveform - waveform.mean()

    peak = waveform.abs().max()

    # PCM float audio can exceed the nominal [-1, 1] range.
    if peak > 1.0:
        waveform = waveform / peak

    return waveform.contiguous()


# ============================================================
# MODEL LOADING
# ============================================================

def load_encoder():
    """Load SpeechBrain's pretrained ECAPA-TDNN speaker encoder."""

    try:
        from speechbrain.inference.classifiers import EncoderClassifier

    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier

        except ImportError as exc:
            raise RuntimeError(
                "SpeechBrain is missing or incompatible.\n"
                "Install it using:\n"
                "python -m pip install speechbrain torch torchaudio numpy"
            ) from exc

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    print("Loading ECAPA-TDNN speaker embedding model...")

    try:
        encoder = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE,
            savedir=str(MODEL_CACHE),
            run_opts={"device": "cpu"},
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not load SpeechBrain model: {exc}"
        ) from exc

    return encoder


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def create_embedding(encoder, waveform: torch.Tensor) -> torch.Tensor:
    """Generate and validate a normalized speaker embedding."""

    if waveform.ndim != 1:
        raise ValueError(
            f"Expected mono 1D audio, got shape {tuple(waveform.shape)}."
        )

    if waveform.numel() == 0:
        raise ValueError("Audio is empty.")

    with torch.inference_mode():
        embedding = encoder.encode_batch(
            waveform.unsqueeze(0)
        )

    embedding = embedding.squeeze().detach().to(
        device="cpu",
        dtype=torch.float32,
    )

    if embedding.ndim != 1 or embedding.numel() == 0:
        raise ValueError(
            f"Unexpected embedding shape: {tuple(embedding.shape)}."
        )

    if not torch.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinite values.")

    norm = torch.linalg.vector_norm(embedding)

    if norm <= 1e-12:
        raise ValueError("Embedding has zero magnitude.")

    return F.normalize(embedding, p=2, dim=0).contiguous()


# ============================================================
# SAFE OUTPUT
# ============================================================

def save_voiceprint(payload: dict) -> None:
    """Write, reload, validate, then atomically replace the output."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = OUTPUT_FILE.with_name(
        OUTPUT_FILE.stem + ".tmp" + OUTPUT_FILE.suffix
    )

    try:
        torch.save(payload, temporary_file)

        # Reload the file before replacing the existing voiceprint.
        check = torch.load(
            temporary_file,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(check, dict):
            raise RuntimeError("Saved payload is not a dictionary.")

        embedding = check.get("embedding")

        if not isinstance(embedding, torch.Tensor):
            raise RuntimeError(
                "Saved payload does not contain an embedding tensor."
            )

        if embedding.ndim != 1 or embedding.numel() == 0:
            raise RuntimeError("Saved embedding has an invalid shape.")

        if not torch.isfinite(embedding).all():
            raise RuntimeError("Saved embedding contains invalid values.")

        if embedding.dtype != torch.float32:
            raise RuntimeError("Saved embedding must be float32.")

        # The temporary file is in the same directory, allowing an
        # atomic replacement on the same filesystem.
        temporary_file.replace(OUTPUT_FILE)

    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print(f"Recordings folder: {SAMPLES_DIR}")
    print(f"Output file:       {OUTPUT_FILE}")

    if not SAMPLES_DIR.is_dir():
        print("\nERROR: Recordings folder not found.")
        print(f"Create this folder and add your recordings:\n{SAMPLES_DIR}")
        return 1

    recordings = sorted(
        path
        for path in SAMPLES_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not recordings:
        print("\nERROR: No supported audio recordings found.")
        print("Supported formats:", ", ".join(sorted(AUDIO_EXTENSIONS)))
        return 1

    print(f"Found {len(recordings)} candidate recording(s).")

    try:
        encoder = load_encoder()

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    embeddings = []
    used_files = []
    skipped = []

    # --------------------------------------------------------
    # Process every recording independently.
    # --------------------------------------------------------

    for index, path in enumerate(recordings, start=1):
        relative = path.relative_to(SAMPLES_DIR)

        try:
            waveform = load_audio(path)
            duration = waveform.numel() / SAMPLE_RATE

            if duration < MIN_DURATION_SECONDS:
                reason = (
                    f"too short ({duration:.2f}s; "
                    f"minimum {MIN_DURATION_SECONDS:.1f}s)"
                )

                skipped.append((str(relative), reason))
                print(f"[{index}/{len(recordings)}] SKIP {relative}: {reason}")
                continue

            embedding = create_embedding(encoder, waveform)

            embeddings.append(embedding)
            used_files.append(str(relative))

            print(
                f"[{index}/{len(recordings)}] OK   "
                f"{relative} ({duration:.1f}s)"
            )

        except Exception as exc:
            reason = str(exc)
            skipped.append((str(relative), reason))

            print(
                f"[{index}/{len(recordings)}] SKIP "
                f"{relative}: {reason}"
            )

    # --------------------------------------------------------
    # Stop if no usable recordings remain.
    # --------------------------------------------------------

    if not embeddings:
        print("\nERROR: No recordings were usable.")
        print("No new voiceprint was written.")
        return 1

    # --------------------------------------------------------
    # Average the normalized embeddings and normalize the centroid.
    # --------------------------------------------------------

    voiceprint = F.normalize(
        torch.stack(embeddings, dim=0).mean(dim=0),
        p=2,
        dim=0,
    ).to(dtype=torch.float32).contiguous()

    if not torch.isfinite(voiceprint).all():
        print("\nERROR: Final voiceprint contains invalid values.")
        return 1

    payload = {
        "embedding": voiceprint,
        "sample_rate": SAMPLE_RATE,
        "model": MODEL_SOURCE,
        "num_recordings": len(used_files),
        "embedding_size": int(voiceprint.numel()),
        "recordings": used_files,
    }

    # --------------------------------------------------------
    # Save and validate the final artifact.
    # --------------------------------------------------------

    try:
        save_voiceprint(payload)

    except Exception as exc:
        print(f"\nERROR: Failed to save or validate voiceprint: {exc}")
        return 1

    print("\n" + "=" * 54)
    print("VOICEPRINT CREATED")
    print("=" * 54)
    print(f"File:               {OUTPUT_FILE}")
    print(f"Recordings used:    {len(used_files)}")
    print(f"Recordings skipped: {len(skipped)}")
    print(f"Embedding size:     {voiceprint.numel()} dimensions")
    print(f"Model:              {MODEL_SOURCE}")

    if skipped:
        print("\nSkipped recordings:")
        for filename, reason in skipped:
            print(f"  - {filename}: {reason}")

    print(
        "\nIMPORTANT: Include only Hasin's voice recordings. "
        "Other speakers can weaken verification accuracy."
    )

    print(
        "\nThe voiceprint file was created successfully. "
        "Live speaker verification still needs to be tested separately."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
