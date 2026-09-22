from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from memory.models import MemoryItem, MemoryType


@dataclass(frozen=True)
class MemoryRetrievalResult:
    """
    Transient evidence describing why a memory item was retrieved.

    MemoryItem remains the durable memory-domain object. Retrieval-specific
    state such as rank and scores belongs here because it is specific to a
    single retrieval operation and must not be persisted with the memory.
    """

    item: MemoryItem
    retrieval_method: str
    rank: int
    retrieval_score: float | None = None
    reranker_score: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.retrieval_method.strip():
            raise ValueError("Memory retrieval method must not be empty.")

        if self.rank <= 0:
            raise ValueError("Memory retrieval rank must be greater than zero.")


class MemoryRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> Sequence[MemoryRetrievalResult]: ...
