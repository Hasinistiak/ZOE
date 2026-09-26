from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np

from .database import (
    memory_database,
)

from .embeddings import (
    embedding_provider,
)

from .models import (
    DurableMemory,
    Memory,
)

from .scorer import (
    score_memory,
)

from .vector_store import (
    vector_store,
)


# ============================================================
# CONFIG
# ============================================================

VECTOR_CANDIDATES = 25

DEFAULT_RESULTS = 5

MAX_RESULTS = 20

MAX_MEMORY_CHARS = 1200

MAX_TOTAL_MEMORY_CHARS = 4000

MIN_SEMANTIC_SIMILARITY = 0.35

DURABLE_CANDIDATES = 25


# ============================================================
# RETRIEVER
# ============================================================

class MemoryRetriever:

    def __init__(
        self,
        database=None,
        embeddings=None,
        vector_store_instance=None,
    ) -> None:

        self.database = (
            database
            or memory_database
        )

        self.embeddings = (
            embeddings
            or embedding_provider
        )

        self.vector_store = (
            vector_store_instance
            or vector_store
        )

    # ========================================================
    # STORE
    # ========================================================

    def store(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: str,
        importance: float = 0.5,
        memory_type: str = "conversation",
        metadata: dict[str, Any] | None = None,
    ) -> int:

        content = str(
            content
        ).strip()

        if not content:

            raise ValueError(
                "Cannot store empty memory."
            )

        session_id = str(
            session_id
        )

        role = str(
            role
        ).strip().lower()

        timestamp = str(
            timestamp
        )

        # ----------------------------------------------------
        # Prevent duplicate messages.
        # ----------------------------------------------------

        existing = (
            self.database.find_message(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=timestamp,
            )
        )

        if existing is not None:

            memory_id = int(
                existing["id"]
            )

            embedding_blob = (
                existing["embedding"]
            )

            dimension = (
                existing["embedding_dimension"]
            )

            if (
                embedding_blob
                and dimension
            ):

                try:

                    embedding = (
                        np.frombuffer(
                            embedding_blob,
                            dtype=np.float32,
                        ).tolist()
                    )

                    if len(embedding) == int(
                        dimension
                    ):

                        self.vector_store.add(
                            memory_id,
                            embedding,
                        )

                except Exception:
                    pass

            return memory_id

        # ----------------------------------------------------
        # Embed.
        # ----------------------------------------------------

        embedding = (
            self.embeddings.embed(
                content
            )
        )

        # ----------------------------------------------------
        # Store in SQLite.
        # ----------------------------------------------------

        memory_id = (
            self.database.insert_memory(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=timestamp,
                importance=importance,
                memory_type=memory_type,
                metadata=metadata,
                embedding=_embedding_to_bytes(
                    embedding
                ),
                embedding_dimension=len(
                    embedding
                ),
            )
        )

        # ----------------------------------------------------
        # Update in-process vector index.
        # ----------------------------------------------------

        self.vector_store.add(
            memory_id,
            embedding,
        )

        return memory_id

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        limit: int = DEFAULT_RESULTS,
        retrieval_context: str = "",
        memory_types: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:

        query = str(
            query
        ).strip()

        if not query:
            return []

        limit = max(
            1,
            min(
                int(limit),
                MAX_RESULTS,
            ),
        )

        allowed_types = None

        if memory_types is not None:

            allowed_types = {
                str(value).strip()
                for value in memory_types
                if str(value).strip()
            }

            if not allowed_types:
                allowed_types = None

        # ----------------------------------------------------
        # Query embedding.
        # ----------------------------------------------------

        if retrieval_context:

            embedding_text = (
                f"{str(retrieval_context).strip()}\n"
                f"CURRENT USER MESSAGE:\n"
                f"{query}"
            )

        else:

            embedding_text = query

        query_embedding = (
            self.embeddings.embed(
                embedding_text
            )
        )

        # ----------------------------------------------------
        # Vector search: conversation memories.
        # ----------------------------------------------------

        conversation_candidates = (
            self.vector_store.search(
                query_embedding,
                limit=VECTOR_CANDIDATES,
            )
        )

        # ----------------------------------------------------
        # Semantic search: durable memories.
        # ----------------------------------------------------

        durable_rows = self.database.get_all_active_durable()

        # Embed and score each durable memory
        durable_texts = []
        for row in durable_rows:
            durable = _row_to_durable(row)
            durable_texts.append(durable.to_semantic_text())

        durable_candidates = []
        if durable_texts:
            durable_embeddings = self.embeddings.embed_batch(durable_texts)
            query_norm = np.asarray(query_embedding, dtype=np.float32)
            query_norm = query_norm / np.linalg.norm(query_norm)

            for i, (row, emb) in enumerate(zip(durable_rows, durable_embeddings)):
                emb_array = np.asarray(emb, dtype=np.float32)
                emb_norm = emb_array / np.linalg.norm(emb_array)
                semantic_score = float(np.dot(query_norm, emb_norm))

                if semantic_score >= MIN_SEMANTIC_SIMILARITY:
                    durable_candidates.append((row, semantic_score))

            # Sort by semantic score and limit
            durable_candidates.sort(key=lambda x: x[1], reverse=True)
            durable_candidates = durable_candidates[:DURABLE_CANDIDATES]

        # ----------------------------------------------------
        # Score all candidates.
        # ----------------------------------------------------

        scored = []

        # Score conversation memories
        for memory_id, semantic_score in conversation_candidates:

            if (
                semantic_score
                < MIN_SEMANTIC_SIMILARITY
            ):

                continue

            row = (
                self.database.get_memory(
                    memory_id
                )
            )

            if row is None:
                continue

            if (
                allowed_types is not None
                and str(row["memory_type"])
                not in allowed_types
            ):

                continue

            memory = _row_to_memory(
                row
            )

            candidate = score_memory(
                memory,
                semantic_score,
            )

            scored.append(
                candidate
            )

        # Score durable memories
        for row, semantic_score in durable_candidates:

            if (
                allowed_types is not None
                and "durable" not in allowed_types
                and row["memory_type"] not in allowed_types
            ):

                continue

            durable = _row_to_durable(row)

            candidate = score_memory(
                durable,
                semantic_score,
            )

            scored.append(
                candidate
            )

        # ----------------------------------------------------
        # Final ranking.
        # ----------------------------------------------------

        scored.sort(
            key=lambda item: (
                item.final_score,
                item.semantic_score,
            ),
            reverse=True,
        )

        results: list[
            dict[str, Any]
        ] = []

        total_chars = 0

        for candidate in scored:

            if isinstance(candidate.memory, DurableMemory):
                content = candidate.memory.to_semantic_text()
            else:
                content = (
                    candidate.memory.content
                )

            if len(content) > MAX_MEMORY_CHARS:

                content = (
                    content[
                        :MAX_MEMORY_CHARS
                    ].rstrip()
                    + "..."
                )

            remaining = (
                MAX_TOTAL_MEMORY_CHARS
                - total_chars
            )

            if remaining <= 0:
                break

            if len(content) > remaining:

                content = (
                    content[
                        :remaining
                    ].rstrip()
                    + "..."
                )

            result = candidate.to_dict()

            result["content"] = content

            results.append(
                result
            )

            total_chars += len(content)

            if len(results) >= limit:
                break

        return results

    # ========================================================
    # BACKFILL
    # ========================================================

    def backfill_embeddings(
        self,
        batch_size: int = 32,
    ) -> int:

        batch_size = max(
            1,
            int(batch_size),
        )

        total = 0

        while True:

            rows = (
                self.database
                .get_unembedded_memories(
                    limit=batch_size
                )
            )

            if not rows:
                break

            texts = [
                str(row["content"]).strip()
                for row in rows
            ]

            embeddings = (
                self.embeddings.embed_batch(
                    texts
                )
            )

            if len(embeddings) != len(rows):

                raise RuntimeError(
                    "Embedding count does not "
                    "match memory count."
                )

            for row, embedding in zip(
                rows,
                embeddings,
            ):

                memory_id = int(
                    row["id"]
                )

                self.database.update_embedding(
                    memory_id=memory_id,
                    embedding=_embedding_to_bytes(
                        embedding
                    ),
                    dimension=len(
                        embedding
                    ),
                )

                self.vector_store.add(
                    memory_id,
                    embedding,
                )

                total += 1

        return total

    # ========================================================
    # STATS
    # ========================================================

    def count(self) -> int:
        return self.database.count()

    def embedded_count(self) -> int:
        return self.database.embedded_count()

    # ========================================================
    # RELOAD
    # ========================================================

    def reload(self) -> None:
        self.vector_store.reload()


# ============================================================
# HELPERS
# ============================================================

def _embedding_to_bytes(
    embedding: list[float],
) -> bytes:

    array = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if array.ndim != 1:

        raise ValueError(
            "Embedding must be one-dimensional."
        )

    if not np.all(
        np.isfinite(array)
    ):

        raise ValueError(
            "Embedding contains invalid values."
        )

    if len(array) == 0:

        raise ValueError(
            "Embedding cannot be empty."
        )

    return array.tobytes()


def _row_to_memory(
    row,
) -> Memory:

    metadata_raw = (
        row["metadata"]
        or "{}"
    )

    try:

        metadata = json.loads(
            metadata_raw
        )

    except Exception:

        metadata = {}

    if not isinstance(
        metadata,
        dict,
    ):

        metadata = {}

    return Memory(
        id=int(
            row["id"]
        ),
        session_id=str(
            row["session_id"]
        ),
        role=str(
            row["role"]
        ),
        content=str(
            row["content"]
        ),
        timestamp=str(
            row["timestamp"]
        ),
        importance=float(
            row["importance"]
        ),
        memory_type=str(
            row["memory_type"]
        ),
        metadata=metadata,
    )


def _row_to_durable(
    row,
) -> DurableMemory:

    metadata_raw = (
        row["metadata"]
        or "{}"
    )

    try:

        metadata = json.loads(
            metadata_raw
        )

    except Exception:

        metadata = {}

    if not isinstance(
        metadata,
        dict,
    ):

        metadata = {}

    return DurableMemory(
        id=int(
            row["id"]
        ),
        subject=str(
            row["subject"]
        ),
        key=str(
            row["memory_key"]
        ),
        value=str(
            row["value"]
        ),
        memory_type=str(
            row["memory_type"]
        ),
        confidence=float(
            row["confidence"]
        ),
        importance=float(
            row["importance"]
        ),
        status=str(
            row["status"]
        ),
        created_at=str(
            row["created_at"]
        ),
        updated_at=str(
            row["updated_at"]
        ),
        valid_from=str(
            row["valid_from"]
        ),
        valid_until=(
            str(row["valid_until"])
            if row["valid_until"] is not None
            else None
        ),
        source_session_id=(
            str(row["source_session_id"])
            if row["source_session_id"] is not None
            else None
        ),
        source_memory_id=(
            int(row["source_memory_id"])
            if row["source_memory_id"] is not None
            else None
        ),
        supersedes_id=(
            int(row["supersedes_id"])
            if row["supersedes_id"] is not None
            else None
        ),
        metadata=metadata,
    )


# ============================================================
# GLOBAL
# ============================================================

memory_retriever = MemoryRetriever()