
from __future__ import annotations

import socket
import threading
import time

import sounddevice as sd


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 5000

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # Signed 16-bit PCM = 2 bytes per sample

BLOCK_SIZE = 512
MAX_BUFFER_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * 2  # 2 seconds

# ============================================================
# AUDIO BUFFER
# ============================================================

audio_buffer = bytearray()
buffer_lock = threading.Lock()

running = True


def receive_audio(client: socket.socket) -> None:
    """Receive raw PCM bytes from the ESP32."""
    global audio_buffer

    while running:
        try:
            data = client.recv(4096)

            if not data:
                break

            with buffer_lock:
                audio_buffer.extend(data)

                # Drop oldest audio if playback falls behind.
                if len(audio_buffer) > MAX_BUFFER_BYTES:
                    overflow = len(audio_buffer) - MAX_BUFFER_BYTES
                    overflow += overflow % SAMPLE_WIDTH
                    del audio_buffer[:overflow]

        except (ConnectionResetError, ConnectionAbortedError, OSError):
            break


def playback_callback(outdata, frames, time_info, status) -> None:
    """Feed incoming PCM audio to the sound device."""
    global audio_buffer

    required_bytes = frames * SAMPLE_WIDTH

    with buffer_lock:
        available = min(len(audio_buffer), required_bytes)
        available -= available % SAMPLE_WIDTH

        if available:
            outdata[:available] = audio_buffer[:available]
            del audio_buffer[:available]

        if available < required_bytes:
            outdata[available:required_bytes] = (
                b"\x00" * (required_bytes - available)
            )


def main() -> None:
    global running, audio_buffer

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((HOST, PORT))
        server.listen(1)

        print("=" * 48)
        print("       ESP32 LIVE MICROPHONE AUDIO")
        print("=" * 48)
        print(f"Listening on {HOST}:{PORT}")
        print(f"Playback: {SAMPLE_RATE} Hz, mono, 16-bit PCM")
        print("Waiting for ESP32 connection...")
        print("Press Ctrl+C to stop.\n")

        with sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype="int16",
            callback=playback_callback,
        ):
            while True:
                client, address = server.accept()
                print(f"\n[CONNECTED] {address[0]}:{address[1]}")

                with buffer_lock:
                    audio_buffer.clear()

                receiver = threading.Thread(
                    target=receive_audio,
                    args=(client,),
                    daemon=True,
                )
                receiver.start()

                try:
                    while receiver.is_alive():
                        time.sleep(0.1)
                finally:
                    client.close()

                    with buffer_lock:
                        audio_buffer.clear()

                    print("[DISCONNECTED] Waiting for ESP32...")

    except KeyboardInterrupt:
        print("\nStopping audio receiver...")

    except OSError as exc:
        print(f"\n[ERROR] {exc}")
        print("Check whether port 5000 is already in use.")

    finally:
        running = False
        server.close()
        print("Audio receiver stopped.")


if __name__ == "__main__":
    main()
