from __future__ import annotations

import threading

import numpy as np

from .database import (
    memory_database,
)


# ============================================================
# VECTOR STORE
# ============================================================

class VectorStore:

    """
    Small in-process cosine-similarity vector index.

    The SQLite database remains the source of truth.

    VectorStore is only an acceleration/index layer.
    """

    def __init__(
        self,
        database=None,
    ) -> None:

        self.database = (
            database
            or memory_database
        )

        self._lock = threading.RLock()

        self._ids = np.array(
            [],
            dtype=np.int64,
        )

        self._vectors = np.empty(
            (0, 0),
            dtype=np.float32,
        )

        self._dimension: int | None = None

        self.reload()

    # ========================================================
    # RELOAD
    # ========================================================

    def reload(self) -> None:

        rows = (
            self.database
            .get_embedded_memories()
        )

        ids: list[int] = []

        vectors: list[np.ndarray] = []

        dimension: int | None = None

        for row in rows:

            blob = row["embedding"]

            row_dimension = (
                row["embedding_dimension"]
            )

            if (
                not blob
                or not row_dimension
            ):
                continue

            try:

                vector = np.frombuffer(
                    blob,
                    dtype=np.float32,
                )

            except Exception:

                continue

            if vector.ndim != 1:
                continue

            if len(vector) != int(
                row_dimension
            ):
                continue

            if dimension is None:
                dimension = int(
                    row_dimension
                )

            if int(row_dimension) != dimension:
                continue

            norm = np.linalg.norm(
                vector
            )

            if (
                not np.isfinite(norm)
                or norm == 0
            ):
                continue

            ids.append(
                int(row["id"])
            )

            vectors.append(
                (
                    vector / norm
                ).astype(
                    np.float32,
                    copy=True,
                )
            )

        with self._lock:

            if not vectors:

                self._ids = np.array(
                    [],
                    dtype=np.int64,
                )

                self._vectors = np.empty(
                    (0, 0),
                    dtype=np.float32,
                )

                self._dimension = None

                return

            self._ids = np.asarray(
                ids,
                dtype=np.int64,
            )

            self._vectors = np.vstack(
                vectors
            ).astype(
                np.float32,
                copy=False,
            )

            self._dimension = dimension

    # ========================================================
    # ADD
    # ========================================================

    def add(
        self,
        memory_id: int,
        embedding: list[float],
    ) -> None:

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if vector.ndim != 1:

            raise ValueError(
                "Embedding must be 1-dimensional."
            )

        if not np.all(
            np.isfinite(vector)
        ):

            raise ValueError(
                "Embedding contains invalid values."
            )

        norm = np.linalg.norm(
            vector
        )

        if (
            not np.isfinite(norm)
            or norm == 0
        ):

            raise ValueError(
                "Embedding vector has zero magnitude."
            )

        vector = (
            vector / norm
        ).astype(
            np.float32
        )

        with self._lock:

            if self._dimension is None:

                self._dimension = len(
                    vector
                )

            if (
                len(vector)
                != self._dimension
            ):

                raise ValueError(
                    "Embedding dimension mismatch."
                )

            existing = np.where(
                self._ids == int(memory_id)
            )[0]

            if len(existing):

                index = int(
                    existing[0]
                )

                self._vectors[index] = vector

                return

            if self._vectors.size == 0:

                self._vectors = vector[
                    None,
                    :,
                ]

            else:

                self._vectors = np.vstack(
                    (
                        self._vectors,
                        vector,
                    )
                )

            self._ids = np.append(
                self._ids,
                int(memory_id),
            )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        embedding: list[float],
        limit: int = 25,
    ) -> list[tuple[int, float]]:

        limit = int(limit)

        if limit <= 0:
            return []

        query = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if query.ndim != 1:
            return []

        if not np.all(
            np.isfinite(query)
        ):
            return []

        with self._lock:

            if self._vectors.size == 0:
                return []

            if (
                self._dimension
                != len(query)
            ):

                return []

            norm = np.linalg.norm(
                query
            )

            if (
                not np.isfinite(norm)
                or norm == 0
            ):

                return []

            query = (
                query / norm
            ).astype(
                np.float32,
                copy=False,
            )

            similarities = (
                self._vectors @ query
            )

            count = min(
                int(limit),
                len(similarities),
            )

            if count <= 0:
                return []

            indices = np.argpartition(
                -similarities,
                count - 1,
            )[:count]

            indices = indices[
                np.argsort(
                    -similarities[indices]
                )
            ]

            return [
                (
                    int(
                        self._ids[index]
                    ),
                    float(
                        similarities[index]
                    ),
                )
                for index in indices
            ]


# ============================================================
# GLOBAL
# ============================================================

vector_store = VectorStore(
    memory_database
)