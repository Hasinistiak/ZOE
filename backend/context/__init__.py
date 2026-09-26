from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# CONTEXT FILES
# ============================================================

CONTEXT_DIR = Path(__file__).resolve().parent

IDENTITY_FILE = CONTEXT_DIR / "Identity.md"
SOUL_FILE = CONTEXT_DIR / "Soul.md"
USER_FILE = CONTEXT_DIR / "User.md"


# ============================================================
# FILE LOADER
# ============================================================

def _read_file(path: Path) -> str:
    """
    Read a context markdown file.

    Missing or unreadable files return an empty string.
    """

    try:
        return path.read_text(
            encoding="utf-8"
        ).strip()

    except (FileNotFoundError, OSError):
        return ""


# ============================================================
# RAW LOADERS
# ============================================================

def get_identity() -> str:
    """
    Load ZOE's identity from Identity.md.

    This function performs a direct file read.

    Session-level caching is handled by ContextEngine.
    """

    return _read_file(IDENTITY_FILE)


def get_soul() -> str:
    """
    Load ZOE's behavioral and personality definition
    from Soul.md.

    This function performs a direct file read.

    Session-level caching is handled by ContextEngine.
    """

    return _read_file(SOUL_FILE)


def get_user_context() -> str:
    """
    Load stable user context from User.md.

    This function performs a direct file read.

    Session-level caching is handled by ContextEngine.
    """

    return _read_file(USER_FILE)


# ============================================================
# CONTEXT PACKAGE
# ============================================================

@dataclass
class ContextPackage:
    """
    Context gathered for a single Brain invocation.

    Static session context:
        identity
        soul
        user

    Dynamic invocation context:
        status
        session
        memories

    Identity, Soul, and User are loaded once by the
    ContextEngine and reused for the lifetime of the engine.

    Memory retrieval is intentionally outside this class.
    The Brain decides whether memory is necessary and,
    if so, supplies the retrieved memories here.
    """

    # --------------------------------------------------------
    # STATIC SESSION CONTEXT
    # --------------------------------------------------------

    identity: str = ""

    soul: str = ""

    user: str = ""

    # --------------------------------------------------------
    # DYNAMIC CONTEXT
    # --------------------------------------------------------

    status: dict[str, Any] = field(
        default_factory=dict
    )

    session: list[dict[str, str]] = field(
        default_factory=list
    )

    memories: list[dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the package into a plain dictionary.
        """

        return {
            "identity": self.identity,
            "soul": self.soul,
            "user": self.user,
            "status": self.status,
            "session": self.session,
            "memories": self.memories,
        }


# ============================================================
# CONTEXT ENGINE
# ============================================================

class ContextEngine:
    """
    Session-level context manager for ZOE.

    The ContextEngine owns the lifecycle of static context:

        Identity.md
        Soul.md
        User.md

    All three are loaded exactly once when the engine is
    initialized.

    They remain cached for the lifetime of this engine.

    Dynamic context is assembled separately for each Brain
    invocation:

        status
        session
        memories

    The ContextEngine does NOT:

        - decide actions
        - decide whether memory is needed
        - retrieve memories
        - search knowledge
        - execute agents
        - execute tools
        - manage conversation logic
        - call an LLM
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self) -> None:
        """
        Create a session-level context engine.

        Identity, Soul, and User are loaded once here.

        They are deliberately NOT re-read on every build().
        """

        self._identity: str = get_identity()
        self._soul: str = get_soul()
        self._user: str = get_user_context()

    # ========================================================
    # MAIN
    # ========================================================

    def build(
        self,
        message: str,
        session: list[dict[str, str]] | None = None,
        status: dict[str, Any] | None = None,
        memories: list[dict[str, Any]] | None = None,
    ) -> ContextPackage:
        """
        Build the context package for one Brain invocation.

        Static context comes from the session-level cache.

        Dynamic context is normalized for this invocation.

        `memories` is optional because memory retrieval is
        controlled by the Brain rather than the ContextEngine.
        """

        if not message or not message.strip():
            raise ValueError(
                "Cannot build context for empty message."
            )

        return ContextPackage(
            identity=self._identity,
            soul=self._soul,
            user=self._user,
            status=self._get_status(status),
            session=self._get_session(session or []),
            memories=self._get_memories(memories or []),
        )

    # ========================================================
    # STATIC CONTEXT
    # ========================================================

    @property
    def identity(self) -> str:
        """
        Cached Identity.md contents.

        No file read occurs here.
        """

        return self._identity

    @property
    def soul(self) -> str:
        """
        Cached Soul.md contents.

        No file read occurs here.
        """

        return self._soul

    @property
    def user(self) -> str:
        """
        Cached User.md contents.

        No file read occurs here.
        """

        return self._user

    # ========================================================
    # EXPLICIT RELOAD
    # ========================================================

    def reload(self) -> None:
        """
        Explicitly reload all session-level static context.

        Normally this should NOT be called during a normal
        conversation.

        It exists for development tools, configuration reloads,
        or starting a deliberately refreshed context session.
        """

        self._identity = get_identity()
        self._soul = get_soul()
        self._user = get_user_context()

    # ========================================================
    # STATUS
    # ========================================================

    @staticmethod
    def _get_status(
        status: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Normalize dynamic runtime status.

        Status is invocation-specific and therefore is NOT cached.
        """

        if not isinstance(status, dict):
            return {}

        return dict(status)

    # ========================================================
    # SESSION
    # ========================================================

    @staticmethod
    def _get_session(
        session: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Normalize conversational history.

        Only the most recent 10 valid messages are retained.

        Session itself remains dynamic and is never cached by
        ContextEngine.
        """

        clean: list[dict[str, str]] = []

        for message in session[-10:]:

            if not isinstance(message, dict):
                continue

            role = (
                message.get(
                    "role",
                    "unknown",
                )
                or "unknown"
            )

            content = (
                message.get(
                    "content",
                    "",
                )
                or ""
            )

            role = str(role).strip()
            content = str(content).strip()

            if not content:
                continue

            clean.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return clean

    # ========================================================
    # MEMORIES
    # ========================================================

    @staticmethod
    def _get_memories(
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Normalize retrieved memory data.

        Memory retrieval itself belongs to the Brain/memory
        subsystem. ContextEngine only packages the result.

        Memories are invocation-specific and are never cached.
        """

        if not isinstance(memories, list):
            return []

        return [
            memory
            for memory in memories
            if isinstance(memory, dict)
        ]


# ============================================================
# FACTORY
# ============================================================

def create_context_engine() -> ContextEngine:
    """
    Create a new session-level ContextEngine.

    Creating a new engine starts a new static-context lifecycle:

        Identity → loaded
        Soul     → loaded
        User     → loaded
    """

    return ContextEngine()


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "ContextPackage",
    "ContextEngine",
    "create_context_engine",
    "get_identity",
    "get_soul",
    "get_user_context",
]