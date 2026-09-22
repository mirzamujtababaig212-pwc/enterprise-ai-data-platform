from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from memory.models import MemoryItem, MemoryType
from memory.retrieval.contracts import MemoryRetrievalResult

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
LEXICAL_CANDIDATE_LIMIT = 10_000

_BM25_K1 = 1.2
_BM25_B = 0.75


class MemoryStore(Protocol):
    async def search(
        self,
        namespace: str,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryItem]: ...


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_PATTERN.findall(text))


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
    ) -> Sequence[MemoryRetrievalResult]:
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

        if not candidates:
            return ()

        tokenized_candidates = [(item, _tokenize(item.content)) for item in candidates]

        non_empty_candidates = [(item, tokens) for item, tokens in tokenized_candidates if tokens]

        if not non_empty_candidates:
            return ()

        document_frequency = Counter(
            token for _, tokens in non_empty_candidates for token in set(tokens)
        )

        document_count = len(non_empty_candidates)

        average_document_length = (
            sum(len(tokens) for _, tokens in non_empty_candidates) / document_count
        )

        if average_document_length == 0.0:
            return ()

        query_terms = set(query_tokens)

        scored: list[tuple[MemoryItem, float]] = []

        for item, tokens in non_empty_candidates:
            term_frequencies = Counter(tokens)
            document_length = len(tokens)

            score = 0.0

            for term in query_terms:
                frequency = term_frequencies.get(term, 0)

                if frequency == 0:
                    continue

                df = document_frequency[term]

                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))

                denominator = frequency + _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * document_length / average_document_length
                )

                score += idf * (frequency * (_BM25_K1 + 1.0) / denominator)

            if score > 0.0:
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

        return tuple(
            MemoryRetrievalResult(
                item=item,
                retrieval_method="lexical.bm25",
                rank=rank,
                retrieval_score=score,
            )
            for rank, (item, score) in enumerate(ordered[:top_k], start=1)
        )
