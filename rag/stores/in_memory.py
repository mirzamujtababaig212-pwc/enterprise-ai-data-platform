from __future__ import annotations
import math
from collections.abc import Sequence
from rag.models import EmbeddedChunk, RetrievalResult


class InMemoryVectorStore:
    """
    Deterministic in-memory vector store for local development and tests.

    This implementation exists behind the VectorStore contract so that
    production vector infrastructure can be introduced later without
    changing the RAG service or retrieval API.
    """

    def __init__(self) -> None:
        self._items: dict[str, EmbeddedChunk] = {}

    async def upsert(
        self,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        for item in chunks:
            self._items[item.chunk.id] = item

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        query = tuple(float(value) for value in embedding)

        results: list[RetrievalResult] = []

        for item in self._items.values():
            score = self._cosine_similarity(
                query,
                item.embedding,
            )

            results.append(
                RetrievalResult(
                    chunk=item.chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding dimensions must match.")

        if not left:
            return 0.0

        dot_product = sum(a * b for a, b in zip(left, right))

        left_norm = math.sqrt(sum(value * value for value in left))

        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return dot_product / (left_norm * right_norm)
