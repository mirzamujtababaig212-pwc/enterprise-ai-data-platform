from __future__ import annotations

from dataclasses import dataclass

from memory.models import MemoryItem
from memory.service import MemoryService


@dataclass(frozen=True)
class MemoryContext:
    working: tuple[MemoryItem, ...]
    semantic: tuple[MemoryItem, ...]
    episodic: tuple[MemoryItem, ...]

    @property
    def all_items(self) -> tuple[MemoryItem, ...]:
        return self.working + self.semantic + self.episodic

    @property
    def is_empty(self) -> bool:
        return not self.all_items


class MemoryContextBuilder:
    """
    Builds a structured context from the agent's memories.

    The builder intentionally depends on MemoryService rather than
    directly accessing a concrete memory store.
    """

    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    async def build(
        self,
        namespace: str,
        *,
        working_limit: int = 5,
        semantic_limit: int = 5,
        episodic_limit: int = 5,
    ) -> MemoryContext:
        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        if working_limit <= 0:
            raise ValueError("working_limit must be greater than zero.")

        if semantic_limit <= 0:
            raise ValueError("semantic_limit must be greater than zero.")

        if episodic_limit <= 0:
            raise ValueError("episodic_limit must be greater than zero.")

        working = await self.memory_service.recall(
            namespace,
            memory_type="working",
            limit=working_limit,
        )

        semantic = await self.memory_service.recall(
            namespace,
            memory_type="semantic",
            limit=semantic_limit,
        )

        episodic = await self.memory_service.recall(
            namespace,
            memory_type="episodic",
            limit=episodic_limit,
        )

        return MemoryContext(
            working=tuple(working),
            semantic=tuple(semantic),
            episodic=tuple(episodic),
        )
