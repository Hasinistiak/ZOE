from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

MEMORY_DB_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "zoe_memory.db"
)


# ============================================================
# DATABASE
# ============================================================

class MemoryDatabase:

    def __init__(
        self,
        path: Path = MEMORY_DB_PATH,
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

        self._initialize()

    # ========================================================
    # CONNECTION
    # ========================================================

    def _connect(
        self,
    ) -> sqlite3.Connection:

        db = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )

        db.row_factory = sqlite3.Row

        db.execute(
            "PRAGMA busy_timeout = 30000"
        )

        db.execute(
            "PRAGMA journal_mode = WAL"
        )

        db.execute(
            "PRAGMA foreign_keys = ON"
        )

        return db

    # ========================================================
    # INITIALIZE
    # ========================================================

    def _initialize(self) -> None:

        with self._lock:

            with self._connect() as db:

                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,

                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,

                        timestamp TEXT NOT NULL,

                        importance REAL NOT NULL
                            DEFAULT 0.5,

                        memory_type TEXT NOT NULL
                            DEFAULT 'conversation',

                        metadata TEXT,

                        embedding BLOB,

                        embedding_dimension INTEGER,

                        created_at TEXT NOT NULL
                    )
                    """
                )

                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS durable_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,

                        subject TEXT NOT NULL,

                        memory_key TEXT NOT NULL,

                        value TEXT NOT NULL,

                        memory_type TEXT NOT NULL
                            DEFAULT 'fact',

                        confidence REAL NOT NULL
                            DEFAULT 1.0,

                        importance REAL NOT NULL
                            DEFAULT 0.5,

                        status TEXT NOT NULL
                            DEFAULT 'active',

                        created_at TEXT NOT NULL,

                        updated_at TEXT NOT NULL,

                        valid_from TEXT NOT NULL,

                        valid_until TEXT,

                        source_session_id TEXT,

                        source_memory_id INTEGER,

                        supersedes_id INTEGER,

                        metadata TEXT,

                        FOREIGN KEY (
                            source_memory_id
                        )
                        REFERENCES memories(id),

                        FOREIGN KEY (
                            supersedes_id
                        )
                        REFERENCES durable_memories(id)
                    )
                    """
                )

                db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_memories_timestamp
                    ON memories(timestamp)
                    """
                )

                db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_memories_session
                    ON memories(session_id)
                    """
                )

                db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_memories_type
                    ON memories(memory_type)
                    """
                )

                db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_durable_subject_key
                    ON durable_memories(
                        subject,
                        memory_key
                    )
                    """
                )

                db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_durable_status
                    ON durable_memories(status)
                    """
                )

                db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_durable_updated
                    ON durable_memories(updated_at)
                    """
                )

                db.commit()

    # ========================================================
    # CONVERSATION MEMORY
    # ========================================================

    def insert_memory(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: str,
        importance: float = 0.5,
        memory_type: str = "conversation",
        metadata: dict[str, Any] | None = None,
        embedding: bytes | None = None,
        embedding_dimension: int | None = None,
    ) -> int:

        content = str(content).strip()

        if not content:
            raise ValueError(
                "Cannot store empty memory."
            )

        importance = max(
            0.0,
            min(
                float(importance),
                1.0,
            ),
        )

        created_at = (
            datetime.now()
            .astimezone()
            .isoformat()
        )

        metadata_json = json.dumps(
            metadata or {},
            ensure_ascii=False,
        )

        with self._lock:

            with self._connect() as db:

                cursor = db.execute(
                    """
                    INSERT INTO memories (
                        session_id,
                        role,
                        content,
                        timestamp,
                        importance,
                        memory_type,
                        metadata,
                        embedding,
                        embedding_dimension,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(session_id),
                        str(role),
                        content,
                        str(timestamp),
                        importance,
                        str(memory_type),
                        metadata_json,
                        embedding,
                        embedding_dimension,
                        created_at,
                    ),
                )

                db.commit()

                return int(
                    cursor.lastrowid
                )

    def find_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: str,
    ) -> sqlite3.Row | None:

        with self._lock:

            with self._connect() as db:

                return db.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE session_id = ?
                      AND role = ?
                      AND content = ?
                      AND timestamp = ?
                    LIMIT 1
                    """,
                    (
                        str(session_id),
                        str(role),
                        str(content),
                        str(timestamp),
                    ),
                ).fetchone()

    def get_memory(
        self,
        memory_id: int,
    ) -> sqlite3.Row | None:

        with self._lock:

            with self._connect() as db:

                return db.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE id = ?
                    """,
                    (int(memory_id),),
                ).fetchone()

    def update_embedding(
        self,
        memory_id: int,
        embedding: bytes,
        dimension: int,
    ) -> None:

        with self._lock:

            with self._connect() as db:

                db.execute(
                    """
                    UPDATE memories
                    SET embedding = ?,
                        embedding_dimension = ?
                    WHERE id = ?
                    """,
                    (
                        embedding,
                        int(dimension),
                        int(memory_id),
                    ),
                )

                db.commit()

    def get_embedded_memories(
        self,
    ) -> list[sqlite3.Row]:

        with self._lock:

            with self._connect() as db:

                return db.execute(
                    """
                    SELECT
                        id,
                        session_id,
                        role,
                        content,
                        timestamp,
                        importance,
                        memory_type,
                        metadata,
                        embedding,
                        embedding_dimension
                    FROM memories
                    WHERE embedding IS NOT NULL
                    ORDER BY id ASC
                    """
                ).fetchall()

    def get_unembedded_memories(
        self,
        limit: int = 100,
    ) -> list[sqlite3.Row]:

        limit = max(
            1,
            int(limit),
        )

        with self._lock:

            with self._connect() as db:

                return db.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE embedding IS NULL
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

    def count(self) -> int:

        with self._lock:

            with self._connect() as db:

                row = db.execute(
                    """
                    SELECT COUNT(*)
                    FROM memories
                    """
                ).fetchone()

                return int(row[0])

    def embedded_count(self) -> int:

        with self._lock:

            with self._connect() as db:

                row = db.execute(
                    """
                    SELECT COUNT(*)
                    FROM memories
                    WHERE embedding IS NOT NULL
                    """
                ).fetchone()

                return int(row[0])

    # ========================================================
    # DURABLE MEMORY
    # ========================================================

    def get_active_durable(
        self,
        subject: str,
        key: str,
    ) -> sqlite3.Row | None:

        with self._lock:

            with self._connect() as db:

                return db.execute(
                    """
                    SELECT *
                    FROM durable_memories
                    WHERE subject = ?
                      AND memory_key = ?
                      AND status = 'active'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (
                        str(subject),
                        str(key),
                    ),
                ).fetchone()

    def get_durable(
        self,
        memory_id: int,
    ) -> sqlite3.Row | None:

        with self._lock:

            with self._connect() as db:

                return db.execute(
                    """
                    SELECT *
                    FROM durable_memories
                    WHERE id = ?
                    """,
                    (int(memory_id),),
                ).fetchone()

    def insert_durable(
        self,
        subject: str,
        key: str,
        value: str,
        memory_type: str,
        confidence: float,
        importance: float,
        created_at: str,
        updated_at: str,
        valid_from: str,
        source_session_id: str | None = None,
        source_memory_id: int | None = None,
        supersedes_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:

        confidence = max(
            0.0,
            min(
                1.0,
                float(confidence),
            ),
        )

        importance = max(
            0.0,
            min(
                1.0,
                float(importance),
            ),
        )

        metadata_json = json.dumps(
            metadata or {},
            ensure_ascii=False,
        )

        with self._lock:

            with self._connect() as db:

                cursor = db.execute(
                    """
                    INSERT INTO durable_memories (
                        subject,
                        memory_key,
                        value,
                        memory_type,
                        confidence,
                        importance,
                        status,
                        created_at,
                        updated_at,
                        valid_from,
                        valid_until,
                        source_session_id,
                        source_memory_id,
                        supersedes_id,
                        metadata
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, 'active',
                        ?, ?, ?, NULL, ?, ?, ?, ?
                    )
                    """,
                    (
                        str(subject),
                        str(key),
                        str(value),
                        str(memory_type),
                        confidence,
                        importance,
                        str(created_at),
                        str(updated_at),
                        str(valid_from),
                        source_session_id,
                        source_memory_id,
                        supersedes_id,
                        metadata_json,
                    ),
                )

                db.commit()

                return int(
                    cursor.lastrowid
                )

    def supersede_durable(
        self,
        memory_id: int,
        timestamp: str,
    ) -> None:

        with self._lock:

            with self._connect() as db:

                db.execute(
                    """
                    UPDATE durable_memories
                    SET
                        status = 'superseded',
                        valid_until = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND status = 'active'
                    """,
                    (
                        str(timestamp),
                        str(timestamp),
                        int(memory_id),
                    ),
                )

                db.commit()

    def delete_durable(
        self,
        memory_id: int,
        timestamp: str,
    ) -> None:

        with self._lock:

            with self._connect() as db:

                db.execute(
                    """
                    UPDATE durable_memories
                    SET
                        status = 'deleted',
                        valid_until = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND status = 'active'
                    """,
                    (
                        str(timestamp),
                        str(timestamp),
                        int(memory_id),
                    ),
                )

                db.commit()

    def update_durable(
        self,
        memory_id: int,
        value: str,
        confidence: float,
        importance: float,
        timestamp: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        metadata_json = json.dumps(
            metadata or {},
            ensure_ascii=False,
        )

        with self._lock:

            with self._connect() as db:

                db.execute(
                    """
                    UPDATE durable_memories
                    SET
                        value = ?,
                        confidence = ?,
                        importance = ?,
                        updated_at = ?,
                        metadata = ?
                    WHERE id = ?
                      AND status = 'active'
                    """,
                    (
                        str(value),
                        max(
                            0.0,
                            min(
                                1.0,
                                float(confidence),
                            ),
                        ),
                        max(
                            0.0,
                            min(
                                1.0,
                                float(importance),
                            ),
                        ),
                        str(timestamp),
                        metadata_json,
                        int(memory_id),
                    ),
                )

                db.commit()

    def get_all_active_durable(
        self,
    ) -> list[sqlite3.Row]:

        with self._lock:

            with self._connect() as db:

                return db.execute(
                    """
                    SELECT *
                    FROM durable_memories
                    WHERE status = 'active'
                    ORDER BY updated_at DESC
                    """
                ).fetchall()

    def search_durable(
        self,
        subject: str | None = None,
        key: str | None = None,
    ) -> list[sqlite3.Row]:

        clauses = [
            "status = 'active'"
        ]

        params: list[Any] = []

        if subject:

            clauses.append(
                "subject = ?"
            )

            params.append(
                str(subject)
            )

        if key:

            clauses.append(
                "memory_key = ?"
            )

            params.append(
                str(key)
            )

        query = f"""
            SELECT *
            FROM durable_memories
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
        """

        with self._lock:

            with self._connect() as db:

                return db.execute(
                    query,
                    params,
                ).fetchall()


# ============================================================
# GLOBAL
# ============================================================

memory_database = MemoryDatabase()