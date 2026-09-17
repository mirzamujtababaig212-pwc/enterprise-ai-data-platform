from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from memory.models import MemoryItem, MemoryType

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
LEXICAL_CANDIDATE_LIMIT = 10_000


class MemoryStore(Protocol):
    async def search(
        self,
        namespace: str,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryItem]: ...


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_PATTERN.findall(text))


class LexicalMemoryRetriever:
    """
    Deterministic query-aware memory retriever.

    The retriever uses token overlap against memory content while delegating
    namespace, memory-type, and expiration filtering to the MemoryStore.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> Sequence[MemoryItem]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = _tokenize(query)

        if not query_tokens:
            return ()

        # Request all available memories in the namespace so ranking is
        # performed over the complete eligible set rather than only the
        # newest persistence-level records.
        candidates = await self.store.search(
            namespace,
            memory_type=memory_type,
            limit=LEXICAL_CANDIDATE_LIMIT,
        )

        scored: list[tuple[MemoryItem, float]] = []

        for item in candidates:
            memory_tokens = _tokenize(item.content)

            if not memory_tokens:
                continue

            overlap = len(query_tokens & memory_tokens)

            if overlap == 0:
                continue

            score = overlap / len(query_tokens)
            scored.append((item, score))

        if not scored:
            return ()

        ordered = sorted(
            scored,
            key=lambda entry: (
                -entry[1],
                -entry[0].created_at.timestamp(),
                entry[0].id,
            ),
        )

        return tuple(item for item, _ in ordered[:top_k])
