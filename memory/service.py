from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import uuid4

from memory.contracts import MemoryStore
from memory.models import MemoryItem, MemoryType


class MemoryService:
    """
    High-level memory service.

    The service owns memory lifecycle semantics while persistence
    remains behind the MemoryStore abstraction.
    """

    def __init__(
        self,
        store: MemoryStore,
    ) -> None:
        self.store = store

    async def remember(
        self,
        content: str,
        *,
        namespace: str,
        memory_type: MemoryType,
        metadata: dict | None = None,
    ) -> MemoryItem:
        if not content.strip():
            raise ValueError("Memory content must not be empty.")

        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        item = MemoryItem(
            id=f"memory-{uuid4()}",
            memory_type=memory_type,
            content=content,
            namespace=namespace,
            created_at=datetime.now(timezone.utc),
            metadata=dict(metadata or {}),
        )

        await self.store.put(item)

        return item

    async def recall(
        self,
        namespace: str,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryItem]:
        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        return await self.store.search(
            namespace,
            memory_type=memory_type,
            limit=limit,
        )

    async def forget(
        self,
        memory_id: str,
    ) -> None:
        if not memory_id.strip():
            raise ValueError("memory_id must not be empty.")

        await self.store.delete(memory_id)
