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


    return _read_file(IDENTITY_FILE)


def get_soul() -> str:


    return _read_file(SOUL_FILE)


def get_user_context() -> str:


    return _read_file(USER_FILE)


# ============================================================
# CONTEXT PACKAGE
# ============================================================

@dataclass
class ContextPackage:

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

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:

        return {
            "identity": self.identity,
            "soul": self.soul,
            "user": self.user,
            "status": self.status,
            "session": self.session,
        }


# ============================================================
# CONTEXT ENGINE
# ============================================================

class ContextEngine:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self) -> None:


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
    ) -> ContextPackage:


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
        )

    # ========================================================
    # STATIC CONTEXT
    # ========================================================

    @property
    def identity(self) -> str:


        return self._identity

    @property
    def soul(self) -> str:


        return self._soul

    @property
    def user(self) -> str:


        return self._user

    # ========================================================
    # EXPLICIT RELOAD
    # ========================================================

    def reload(self) -> None:


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


        clean: list[dict[str, str]] = []

        if not isinstance(session, list):
            return []

        for message in session:

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

        # Keep the 10 most recent VALID messages.
        return clean[-10:]


# ============================================================
# FACTORY
# ============================================================

def create_context_engine() -> ContextEngine:


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