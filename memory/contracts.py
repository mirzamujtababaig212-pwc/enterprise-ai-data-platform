from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memory.models import MemoryItem, MemoryType
from rag.models import EmbeddingResult


class MemoryStore(Protocol):
    """
    Persistence abstraction for memory items.
    """

    async def put(
        self,
        item: MemoryItem,
    ) -> None: ...

    async def get(
        self,
        memory_id: str,
    ) -> MemoryItem | None: ...

    async def search(
        self,
        namespace: str,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryItem]: ...

    async def delete(
        self,
        memory_id: str,
    ) -> None: ...


class MemoryEmbeddingStore(Protocol):
    """
    Persistence abstraction for memory embeddings.
    """

    async def put(
        self,
        memory_id: str,
        embedding: EmbeddingResult,
    ) -> None: ...

    async def get(
        self,
        memory_id: str,
    ) -> EmbeddingResult | None: ...

    async def delete(
        self,
        memory_id: str,
    ) -> None: ...


class MemoryServiceProtocol(Protocol):
    """
    High-level memory service abstraction.
    """

    async def remember(
        self,
        content: str,
        *,
        namespace: str,
        memory_type: MemoryType,
        metadata: dict | None = None,
        retention_seconds: int | None = None,
    ) -> MemoryItem: ...

    async def recall(
        self,
        namespace: str,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryItem]: ...

    async def forget(
        self,
        memory_id: str,
    ) -> None: ...
