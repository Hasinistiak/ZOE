from ollama import chat
import time
import re

MODEL = "qwen3:0.6b"

TESTS = [
    ("Hi", False),
    ("Good morning", False),
    ("What's the weather today?", False),
    ("What time is it?", False),
    ("What's on my calendar today?", False),
    ("Set a reminder for 6 PM", False),
    ("Play some music", False),
    ("Who is Real Madrid playing next?", False),
    ("Explain virtual memory", False),
    ("What is Python?", False),

    ("What was that programming language I told you I was learning?", True),
    ("What GPU was I considering buying?", True),
    ("What's my usual wake-up time?", True),
    ("What did I call my AI assistant?", True),
    ("What project was I working on yesterday?", True),
    ("Do you remember what phone I have?", True),
    ("What music do I usually listen to?", True),
    ("What did I say about my PC?", True),
    ("What name did I give the coding assistant project?", True),
    ("What were the specs I told you about my computer?", True),

    ("Do you remember?", True),
    ("Remember what we discussed earlier?", True),
    ("Can you remind me what I said?", True),
    ("What did I just tell you?", False),
    ("What did I say about this?", True),
    ("Tell me about my project", True),
    ("Tell me about Python", False),
    ("How did we configure this?", True),
    ("Why did we choose this model?", True),
    ("Which model are you using?", False),
]

SYSTEM = """You are ZOE's memory retrieval gate.

Determine whether the user's message requires persistent memory about the user.

Return exactly one word:
true
or
false

true = the answer requires information from previous conversations, saved user
facts, preferences, previous decisions, or previous project context.

false = the answer does not require persistent memory.

Current conversation context is NOT persistent memory.

Do not explain.
Do not answer the user.
Do not use JSON.
Return ONLY true or false.

 /no_think
"""


def classify(message: str):
    start = time.perf_counter()

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": message},
        ],
        options={
            "temperature": 0,
            "num_predict": 1,
        },
    )

    elapsed = (time.perf_counter() - start) * 1000

    raw = response["message"]["content"].strip().lower()

    # Remove Qwen thinking tags if they somehow appear.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Extract the decision.
    if re.search(r"\btrue\b", raw):
        result = True
    elif re.search(r"\bfalse\b", raw):
        result = False
    else:
        result = None

    return result, elapsed, raw


correct = 0
total_time = 0

print("=" * 70)
print("ZOE MEMORY GATE — QWEN3 0.6B")
print("=" * 70)

for question, expected in TESTS:
    result, ms, raw = classify(question)

    ok = result == expected

    if ok:
        correct += 1

    total_time += ms

    print(
        f"{'✓' if ok else '✗'} "
        f"{ms:7.0f} ms | "
        f"expected={str(expected):5} | "
        f"got={str(result):5} | "
        f"{question}"
    )

    if result is None:
        print(f"         RAW: {raw!r}")

accuracy = correct / len(TESTS) * 100
avg_ms = total_time / len(TESTS)

print("=" * 70)
print(f"Accuracy : {accuracy:.1f}% ({correct}/{len(TESTS)})")
print(f"Avg time : {avg_ms:.0f} ms")
print("=" * 70)