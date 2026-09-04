from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from memory.models import MemoryItem, MemoryType


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    async def put(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    async def get(self, memory_id: str) -> MemoryItem | None:
        item = self._items.get(memory_id)

        if item is None:
            return None

        if self._is_expired(item):
            return None

        return item

    async def search(
        self,
        namespace: str,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryItem]:
        if limit <= 0:
            return []

        results = [
            item
            for item in self._items.values()
            if item.namespace == namespace
            and (memory_type is None or item.memory_type == memory_type)
            and not self._is_expired(item)
        ]

        results.sort(key=lambda item: item.created_at, reverse=True)

        return results[:limit]

    async def delete(self, memory_id: str) -> None:
        self._items.pop(memory_id, None)

    @staticmethod
    def _is_expired(item: MemoryItem) -> bool:
        if item.expires_at is None:
            return False

        now = datetime.now(timezone.utc)

        return item.expires_at <= now
