from __future__ import annotations

from collections.abc import Mapping, Sequence

import faiss
import numpy as np

from rag.models import EmbeddedChunk, RetrievalResult


class FAISSVectorStore:
    """
    In-process FAISS-backed vector store.

    This backend intentionally keeps chunk metadata and embedding identity
    alongside the FAISS index so it preserves the VectorStore contract and
    the deterministic semantics of the in-memory implementation.

    Persistence is intentionally out of scope for this first slice.
    """

    def __init__(self) -> None:
        self._items: dict[str, EmbeddedChunk] = {}
        self._index: faiss.Index | None = None
        self._dimension: int | None = None
        self._chunk_ids: list[str] = []

    async def upsert(
        self,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        if not chunks:
            return

        dimension = len(chunks[0].embedding)

        if dimension == 0:
            raise ValueError("Embeddings must not be empty.")

        for item in chunks:
            if len(item.embedding) != dimension:
                raise ValueError("All embeddings must have the same dimension.")

        if self._dimension is not None and dimension != self._dimension:
            raise ValueError(
                "Embedding dimensions must match: " f"expected {self._dimension}, got {dimension}."
            )

        for item in chunks:
            self._items[item.chunk.id] = item

        self._dimension = dimension
        self._rebuild_index()

    async def delete_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> None:
        changed = False

        for chunk_id in chunk_ids:
            if chunk_id in self._items:
                del self._items[chunk_id]
                changed = True

        if changed:
            if self._items:
                self._rebuild_index()
            else:
                self._index = None
                self._dimension = None
                self._chunk_ids = []

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        query = tuple(float(value) for value in embedding)

        if not query:
            return []

        if not self._items:
            return []

        if self._dimension is None:
            raise RuntimeError("FAISS vector-store dimension is not initialized.")

        if len(query) != self._dimension:
            raise ValueError(
                "Embedding dimensions must match: " f"expected {self._dimension}, got {len(query)}."
            )

        if self._index is None:
            raise RuntimeError("FAISS vector-store index is not initialized.")

        query_vector = self._normalize(query).reshape(1, -1)

        distances, indices = self._index.search(
            query_vector,
            len(self._chunk_ids),
        )

        results: list[RetrievalResult] = []

        for score, index_position in zip(distances[0], indices[0]):
            if index_position < 0:
                continue

            chunk_id = self._chunk_ids[int(index_position)]
            item = self._items[chunk_id]

            if metadata_filter is not None:
                if any(
                    item.chunk.metadata.get(key) != value for key, value in metadata_filter.items()
                ):
                    continue

            results.append(
                RetrievalResult(
                    chunk=item.chunk,
                    score=float(score),
                    embedding_identity=item.embedding_identity,
                )
            )

        results.sort(
            key=lambda result: (-result.score, result.chunk.id),
        )

        return results[:top_k]

    def _rebuild_index(self) -> None:
        if not self._items:
            self._index = None
            self._dimension = None
            self._chunk_ids = []
            return

        if self._dimension is None:
            raise RuntimeError("FAISS vector-store dimension is not initialized.")

        self._chunk_ids = sorted(self._items)

        vectors = np.asarray(
            [self._normalize(self._items[chunk_id].embedding) for chunk_id in self._chunk_ids],
            dtype=np.float32,
        )

        index = faiss.IndexFlatIP(self._dimension)
        index.add(vectors)

        self._index = index

    @staticmethod
    def _normalize(values: Sequence[float]) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32)

        norm = float(np.linalg.norm(vector))

        if norm == 0.0:
            return vector

        return vector / norm
