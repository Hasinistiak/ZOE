from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from typing import Any

from backend.memory.chatlog import (
    create_chatlog,
    add_message,
    close_chatlog,
    save_chatlog,
    submit_for_memory,
)


# ============================================================
# CONFIG
# ============================================================

SESSION_TIMEOUT_SECONDS = 30 * 60


# ============================================================
# SESSION MANAGER
# ============================================================

class SessionManager:
    """
    Manages ZOE's active conversational session.

    Session history contains ONLY:

        - user messages
        - ZOE's conversational replies

    It does NOT contain:

        - long-term memory retrievals
        - memory-gate decisions
        - agent/tool results
        - runtime command results
        - system prompts
        - identity/soul/user context
        - internal reasoning
        - execution metadata
        - background notifications

    Background/proactive notifications must use
    add_notification() and are deliberately excluded from
    conversational session history and conversational memory
    submission.
    """

    def __init__(
        self,
        timeout_seconds: float = SESSION_TIMEOUT_SECONDS,
    ) -> None:

        self.timeout_seconds = float(
            timeout_seconds
        )

        self._session: dict[str, Any] | None = None

        self._last_activity = 0.0

        self._lock = threading.RLock()

        # The user message awaiting its conversational
        # assistant response.
        self._pending_user_message: str | None = None

    # ========================================================
    # SESSION
    # ========================================================

    def _new_session(self) -> None:

        session_id = str(
            uuid.uuid4()
        )

        started_at = (
            datetime.now()
            .astimezone()
            .isoformat()
        )

        self._session = create_chatlog(
            session_id=session_id,
            started_at=started_at,
        )

        self._last_activity = time.monotonic()

        self._pending_user_message = None

        save_chatlog(
            self._session
        )

        print(
            "[ZOE SESSION] New session: "
            f"{session_id[:8]}"
        )

    # ========================================================
    # EXPIRATION
    # ========================================================

    def _expired(self) -> bool:

        if self._session is None:
            return True

        elapsed = (
            time.monotonic()
            - self._last_activity
        )

        return elapsed >= self.timeout_seconds

    # ========================================================
    # GET SESSION
    # ========================================================

    def get_session(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            if (
                self._session is None
                or self._expired()
            ):

                if self._session is not None:
                    self.close()

                self._new_session()

            self._last_activity = time.monotonic()

            return self._session

    # ========================================================
    # USER MESSAGE
    # ========================================================

    def add_user_message(
        self,
        content: str,
    ) -> None:
        """
        Add a genuine user conversational message.

        This message becomes the pending user turn that will
        later be paired with ZOE's conversational response
        for memory submission.
        """

        content = str(
            content
        ).strip()

        if not content:
            return

        session = self.get_session()

        add_message(
            session,
            "user",
            content,
        )

        with self._lock:

            self._pending_user_message = content

            self._last_activity = time.monotonic()

    # ========================================================
    # ASSISTANT MESSAGE
    # ========================================================

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        """
        Add a genuine conversational ZOE reply.

        Only this type of assistant output becomes part of
        conversational session history.

        The completed user/assistant pair is submitted to
        MemoryManager.
        """

        content = str(
            content
        ).strip()

        if not content:
            return

        session = self.get_session()

        add_message(
            session,
            "assistant",
            content,
        )

        with self._lock:

            user_message = (
                self._pending_user_message
            )

            self._pending_user_message = None

            self._last_activity = time.monotonic()

        # Submit ONLY an actual conversational turn.
        if user_message:

            submit_for_memory(
                chatlog=session,
                user_content=user_message,
                assistant_content=content,
            )

    # ========================================================
    # BACKGROUND NOTIFICATION
    # ========================================================

    def add_notification(
        self,
        content: str,
    ) -> None:
        """
        Handle a proactive/background ZOE notification.

        Notifications intentionally DO NOT:

            - enter session history
            - become assistant messages
            - get paired with a pending user message
            - get submitted to conversational memory

        The runtime/UI/TTS layer is responsible for actually
        delivering the notification.
        """

        content = str(
            content
        ).strip()

        if not content:
            return

        with self._lock:

            # Notification activity should not mutate the
            # conversational session timeout.
            #
            # A background reminder should not keep a
            # conversational session alive indefinitely.
            pass

    # ========================================================
    # RECENT SESSION
    # ========================================================

    def get_recent_messages(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        if limit <= 0:
            return []

        session = self.get_session()

        messages = session.get(
            "messages",
            [],
        )

        if not isinstance(
            messages,
            list,
        ):
            return []

        # Defensive filtering:
        #
        # Session context is strictly conversational.
        # Even if another caller accidentally inserts an
        # unsupported role into the chatlog, it must not
        # escape through this API.
        conversational_messages: list[
            dict[str, Any]
        ] = []

        for message in messages:

            if not isinstance(
                message,
                dict,
            ):
                continue

            role = str(
                message.get(
                    "role",
                    "",
                )
                or ""
            ).strip().lower()

            if role not in {
                "user",
                "assistant",
            }:
                continue

            content = str(
                message.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            if not content:
                continue

            conversational_messages.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": message.get(
                        "timestamp"
                    ),
                }
            )

        return conversational_messages[-limit:]

    # ========================================================
    # CONTEXT
    # ========================================================

    def get_context(
        self,
        limit: int = 12,
    ) -> str:
        """
        Return human-readable conversational history.

        Only user and assistant messages are exposed.
        """

        messages = self.get_recent_messages(
            limit=limit
        )

        if not messages:
            return ""

        lines: list[str] = []

        for message in messages:

            role = str(
                message.get(
                    "role",
                    "",
                )
                or ""
            ).strip().lower()

            content = str(
                message.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            if not content:
                continue

            if role == "user":

                prefix = "User"

            elif role == "assistant":

                prefix = "ZOE"

            else:

                continue

            lines.append(
                f"{prefix}: {content}"
            )

        return "\n".join(lines)

    # ========================================================
    # NEW SESSION
    # ========================================================

    def new_session(self) -> None:

        with self._lock:

            if self._session is not None:
                self.close()

            self._new_session()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        with self._lock:

            if self._session is None:
                return

            try:

                close_chatlog(
                    self._session
                )

            finally:

                self._session = None

                self._last_activity = 0.0

                self._pending_user_message = None

    # ========================================================
    # INFO
    # ========================================================

    def get_session_id(
        self,
    ) -> str | None:

        with self._lock:

            if (
                self._session is None
                or self._expired()
            ):
                return None

            return self._session.get(
                "session_id"
            )

    # ========================================================
    # ACTIVE
    # ========================================================

    def is_active(self) -> bool:

        with self._lock:

            return (
                self._session is not None
                and not self._expired()
            )


# ============================================================
# GLOBAL
# ============================================================

session_manager = SessionManager()