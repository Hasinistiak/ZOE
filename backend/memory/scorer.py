from __future__ import annotations

from datetime import datetime, timezone

from .models import (
    DurableMemory,
    Memory,
    MemoryCandidate,
    MemoryType,
)


# ============================================================
# CONFIG
# ============================================================

SEMANTIC_WEIGHT = 0.70
RECENCY_WEIGHT = 0.15
IMPORTANCE_WEIGHT = 0.15

RECENCY_HALF_LIFE_DAYS = 30.0

DURABLE_AUTHORITY_BOOST = 0.25
DURABLE_CONFIDENCE_WEIGHT = 0.10


# ============================================================
# SCORE
# ============================================================

def score_memory(
    memory: MemoryType,
    semantic_score: float,
) -> MemoryCandidate:

    semantic = _normalize_similarity(
        semantic_score
    )

    if isinstance(memory, DurableMemory):
        recency = _recency_score(
            memory.get_timestamp()
        )
        importance = max(
            0.0,
            min(
                float(memory.get_importance()),
                1.0,
            ),
        )
        confidence = max(
            0.0,
            min(
                float(memory.get_confidence()),
                1.0,
            ),
        )
        authority_boost = DURABLE_AUTHORITY_BOOST

        final = (
            semantic * SEMANTIC_WEIGHT
            + recency * RECENCY_WEIGHT
            + importance * IMPORTANCE_WEIGHT
            + confidence * DURABLE_CONFIDENCE_WEIGHT
            + authority_boost
        )

        return MemoryCandidate(
            memory=memory,
            semantic_score=semantic,
            recency_score=recency,
            importance_score=importance,
            confidence_score=confidence,
            authority_boost=authority_boost,
            final_score=final,
        )

    else:
        recency = _recency_score(
            memory.timestamp
        )

        importance = max(
            0.0,
            min(
                float(memory.importance),
                1.0,
            ),
        )

        final = (
            semantic * SEMANTIC_WEIGHT
            + recency * RECENCY_WEIGHT
            + importance * IMPORTANCE_WEIGHT
        )

        return MemoryCandidate(
            memory=memory,
            semantic_score=semantic,
            recency_score=recency,
            importance_score=importance,
            confidence_score=0.0,
            authority_boost=0.0,
            final_score=final,
        )


# ============================================================
# SIMILARITY
# ============================================================

def _normalize_similarity(
    score: float,
) -> float:
    """
    Normalize cosine similarity to [0, 1].

    VectorStore already calculates cosine similarity.

    Do NOT shift [-1, 1] -> [0, 1] here because doing so
    would make a cosine similarity of 0.0 become 0.5 and
    would distort the retrieval threshold.

    Negative similarity is treated as irrelevant.
    """

    return max(
        0.0,
        min(
            float(score),
            1.0,
        ),
    )


# ============================================================
# RECENCY
# ============================================================

def _recency_score(
    timestamp: str,
) -> float:

    try:

        value = datetime.fromisoformat(
            str(timestamp)
        )

        if value.tzinfo is None:

            value = value.replace(
                tzinfo=timezone.utc
            )

        now = datetime.now(
            value.tzinfo
        )

        age_seconds = max(
            0.0,
            (
                now - value
            ).total_seconds(),
        )

        age_days = (
            age_seconds / 86400.0
        )

        return 0.5 ** (
            age_days
            / RECENCY_HALF_LIFE_DAYS
        )

    except Exception:

        return 0.0