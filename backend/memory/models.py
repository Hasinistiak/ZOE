from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


# ============================================================
# RAW MEMORY
# ============================================================

@dataclass(slots=True)
class Memory:

    id: int

    session_id: str

    role: str

    content: str

    timestamp: str

    importance: float

    memory_type: str

    metadata: dict[str, Any]

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
        }


# ============================================================
# DURABLE MEMORY
# ============================================================

@dataclass(slots=True)
class DurableMemory:

    id: int

    subject: str

    key: str

    value: str

    memory_type: str

    confidence: float

    importance: float

    status: str

    created_at: str

    updated_at: str

    valid_from: str

    valid_until: str | None

    source_session_id: str | None

    source_memory_id: int | None

    supersedes_id: int | None

    metadata: dict[str, Any]

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "id": self.id,
            "subject": self.subject,
            "key": self.key,
            "value": self.value,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "importance": self.importance,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "source_session_id": self.source_session_id,
            "source_memory_id": self.source_memory_id,
            "supersedes_id": self.supersedes_id,
            "metadata": self.metadata,
        }

    def to_semantic_text(self) -> str:
        """Convert durable memory to semantic text for embedding/search."""
        return f"{self.subject}.{self.key} = {self.value}"

    def get_timestamp(self) -> str:
        """Get the most relevant timestamp for recency scoring."""
        return self.updated_at

    def get_importance(self) -> float:
        """Get importance score for ranking."""
        return self.importance

    def get_confidence(self) -> float:
        """Get confidence score for authority weighting."""
        return self.confidence


# ============================================================
# RETRIEVAL CANDIDATE
# ============================================================

MemoryType = Union[Memory, DurableMemory]


@dataclass(slots=True)
class MemoryCandidate:

    memory: MemoryType

    semantic_score: float

    recency_score: float

    importance_score: float

    confidence_score: float

    authority_boost: float

    final_score: float

    def to_dict(
        self,
    ) -> dict[str, Any]:

        base = {
            "id": self.memory.id,
            "semantic_score": round(self.semantic_score, 4),
            "recency_score": round(self.recency_score, 4),
            "importance_score": round(self.importance_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "authority_boost": round(self.authority_boost, 4),
            "final_score": round(self.final_score, 4),
        }

        if isinstance(self.memory, DurableMemory):
            base.update({
                "source": "durable",
                "subject": self.memory.subject,
                "key": self.memory.key,
                "value": self.memory.value,
                "memory_type": self.memory.memory_type,
                "confidence": self.memory.confidence,
                "importance": self.memory.importance,
                "status": self.memory.status,
                "created_at": self.memory.created_at,
                "updated_at": self.memory.updated_at,
                "metadata": self.memory.metadata,
            })
        else:
            base.update({
                "source": "conversation",
                "session_id": self.memory.session_id,
                "role": self.memory.role,
                "content": self.memory.content,
                "timestamp": self.memory.timestamp,
                "importance": round(self.memory.importance, 4),
                "memory_type": self.memory.memory_type,
                "metadata": self.memory.metadata,
            })

        return base