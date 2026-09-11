from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
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

    async def delete_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> None:
        for chunk_id in chunk_ids:
            self._items.pop(chunk_id, None)

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        query = tuple(float(value) for value in embedding)

        if self._items:
            stored_dimension = len(next(iter(self._items.values())).embedding)

            if len(query) != stored_dimension:
                raise ValueError(
                    "Embedding dimensions must match: "
                    f"expected {stored_dimension}, got {len(query)}."
                )

        results: list[RetrievalResult] = []

        for item in self._items.values():
            if metadata_filter is not None:
                if any(
                    item.chunk.metadata.get(key) != value for key, value in metadata_filter.items()
                ):
                    continue

            score = self._cosine_similarity(query, item.embedding)

            results.append(
                RetrievalResult(
                    chunk=item.chunk,
                    score=score,
                    embedding_identity=item.embedding_identity,
                )
            )

        results.sort(
            key=lambda result: (-result.score, result.chunk.id),
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
