from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memory.models import MemoryItem, MemoryType


class MemoryRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> Sequence[MemoryItem]: ...
