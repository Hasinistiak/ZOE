
from __future__ import annotations

import os
import time

from google import genai
from dotenv import load_dotenv


load_dotenv()

# ============================================================
# CONFIG
# ============================================================

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
]

PROMPT = """
You are the reasoning core of a personal AI assistant.
Analyze this request and respond with a concise answer:

The user wants to create a Python desktop application that monitors
system resources and alerts them when CPU temperature becomes too high.
What are the main components the application needs?
"""

ROUNDS = 3


# ============================================================
# CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)


# ============================================================
# BENCHMARK
# ============================================================

print("=" * 70)
print("              GEMINI SPEED BENCHMARK")
print("=" * 70)

print(f"\nRounds per model: {ROUNDS}")

for model in MODELS:
    print("\n" + "-" * 70)
    print(f"MODEL: {model}")
    print("-" * 70)

    times: list[float] = []

    for round_number in range(1, ROUNDS + 1):
        print(f"[{round_number}/{ROUNDS}] Testing...", end=" ", flush=True)

        start = time.perf_counter()

        try:
            response = client.models.generate_content(
                model=model,
                contents=PROMPT,
            )

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            text = response.text or ""

            print(f"{elapsed:.2f}s")

            if round_number == 1:
                print(f"Response: {text[:250].replace(chr(10), ' ')}")

        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"FAILED after {elapsed:.2f}s")
            print(f"Error: {exc}")

    if times:
        average = sum(times) / len(times)
        fastest = min(times)
        slowest = max(times)

        print("\nResults:")
        print(f"  Average : {average:.2f}s")
        print(f"  Fastest : {fastest:.2f}s")
        print(f"  Slowest : {slowest:.2f}s")


print("\n" + "=" * 70)
print("Benchmark complete.")
print("=" * 70)
