from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# CONFIG
# ============================================================

CHAT_HISTORY_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "chat_history"
)


# ============================================================
# TIME
# ============================================================

def _now() -> str:
    """
    Return the current local timestamp as ISO-8601.
    """

    return (
        datetime.now()
        .astimezone()
        .isoformat()
    )


# ============================================================
# CREATE
# ============================================================

def create_chatlog(
    session_id: str,
    started_at: str,
) -> dict[str, Any]:
    """
    Create a new in-memory chatlog structure.

    Chatlogs are the permanent transcript of conversations.

    They are separate from ZOE's long-term durable knowledge.
    """

    return {
        "session_id": str(session_id),
        "started_at": str(started_at),
        "ended_at": None,
        "message_count": 0,
        "messages": [],
    }


# ============================================================
# SAVE
# ============================================================

def save_chatlog(
    chatlog: dict[str, Any],
) -> None:
    """
    Persist the complete chatlog atomically.
    """

    started_at = str(
        chatlog.get(
            "started_at",
            _now(),
        )
    )

    session_id = str(
        chatlog.get(
            "session_id",
            "unknown",
        )
    )

    try:
        dt = datetime.fromisoformat(
            started_at
        )

        date_dir = (
            CHAT_HISTORY_DIR
            / dt.strftime("%Y-%m")
        )

        filename = (
            f"{dt.strftime('%Y-%m-%d_%H-%M-%S')}"
            f"_{session_id[:8]}.json"
        )

    except Exception:
        date_dir = CHAT_HISTORY_DIR

        filename = (
            f"{session_id}.json"
        )

    date_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = date_dir / filename

    temp_path = path.with_name(
        f".{path.name}.tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chatlog,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.flush()

    temp_path.replace(path)


# ============================================================
# ADD MESSAGE
# ============================================================

def add_message(
    chatlog: dict[str, Any],
    role: str,
    content: str,
) -> None:
    """
    Add one message to the permanent conversation transcript.

    IMPORTANT:

    This function does not directly perform memory retrieval
    or durable-memory learning.

    Long-term memory processing is handled by MemoryManager.
    """

    content = str(
        content
    ).strip()

    if not content:
        return

    role = str(
        role
    ).strip().lower()

    if not role:
        return

    timestamp = _now()

    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp,
    }

    messages = chatlog.setdefault(
        "messages",
        [],
    )

    messages.append(message)

    chatlog["message_count"] = len(
        messages
    )

    save_chatlog(chatlog)


# ============================================================
# ADD USER MESSAGE
# ============================================================

def add_user_message(
    chatlog: dict[str, Any],
    content: str,
) -> None:
    """
    Convenience wrapper for user messages.
    """

    add_message(
        chatlog=chatlog,
        role="user",
        content=content,
    )


# ============================================================
# ADD ASSISTANT MESSAGE
# ============================================================

def add_assistant_message(
    chatlog: dict[str, Any],
    content: str,
) -> None:
    """
    Convenience wrapper for assistant messages.
    """

    add_message(
        chatlog=chatlog,
        role="assistant",
        content=content,
    )


# ============================================================
# CLOSE
# ============================================================

def close_chatlog(
    chatlog: dict[str, Any],
) -> None:
    """
    Mark the conversation as finished and persist it.
    """

    chatlog["ended_at"] = _now()

    save_chatlog(chatlog)


# ============================================================
# MEMORY INTEGRATION
# ============================================================

def submit_for_memory(
    chatlog: dict[str, Any],
    user_content: str,
    assistant_content: str | None = None,
) -> None:
    """
    Submit a completed conversational turn to MemoryManager.

    MemoryManager handles:

        - raw conversational memory
        - embeddings
        - vector indexing
        - durable-memory analysis
        - ADD / UPDATE / DELETE / IGNORE

    The manager performs its work asynchronously so the
    conversational response path is not blocked.
    """

    user_content = str(
        user_content
    ).strip()

    if not user_content:
        return

    try:
        from backend.memory.manager import (
            memory_manager,
        )

    except Exception as exc:

        print(
            "[MEMORY] Memory manager unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

        return

    try:

        memory_manager.submit_turn(
            session_id=str(
                chatlog.get(
                    "session_id",
                    "",
                )
            ),
            user_message=user_content,
            assistant_message=(
                str(assistant_content).strip()
                if assistant_content
                else None
            ),
        )

    except Exception as exc:

        print(
            "[MEMORY] Failed to submit turn: "
            f"{type(exc).__name__}: {exc}"
        )