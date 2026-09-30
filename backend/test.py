from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MEMORY
# ============================================================

from backend.memory import memory_retriever


# ============================================================
# CONFIG
# ============================================================

MAX_MEMORIES = 5


# ============================================================
# MEMORY QUERY
# ============================================================

def query_memory(query: str) -> list[dict[str, Any]]:
    """
    Query ZOE's long-term memory directly.
    """

    query = query.strip()

    if not query:
        return []

    memories = memory_retriever.retrieve(
        query=query,
        limit=MAX_MEMORIES,
        retrieval_context="",
    )

    if not isinstance(memories, list):
        return []

    return [
        dict(memory)
        for memory in memories
        if isinstance(memory, dict)
    ]


# ============================================================
# DISPLAY
# ============================================================

def print_memories(
    query: str,
    memories: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 70)
    print("ZOE MEMORY QUERY")
    print("=" * 70)

    print(f"\nQuery:\n{query}")

    print(f"\nRetrieved: {len(memories)}")

    if not memories:
        print("\nNo relevant memories found.")
        print()
        return

    print()

    for index, memory in enumerate(memories, start=1):

        print("-" * 70)
        print(f"MEMORY #{index}")
        print("-" * 70)

        print(
            json.dumps(
                memory,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    print()


# ============================================================
# CLI
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("ZOE LONG-TERM MEMORY TEST")
    print("=" * 70)
    print()
    print("Type a query to search memory.")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        try:
            query = input("MEMORY QUERY: ").strip()

        except KeyboardInterrupt:
            print("\n")
            break

        except EOFError:
            print()
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        try:

            memories = query_memory(query)

            print_memories(
                query=query,
                memories=memories,
            )

        except Exception as exc:

            print()
            print("=" * 70)
            print("MEMORY ERROR")
            print("=" * 70)
            print(
                f"{type(exc).__name__}: {exc}"
            )
            print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()