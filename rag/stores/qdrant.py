from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from rag.models import DocumentChunk, EmbeddedChunk, RetrievalResult


class QdrantVectorStore:
    """
    Qdrant-backed implementation of the RAG VectorStore contract.

    The application-level chunk ID remains authoritative. Qdrant receives
    a deterministic UUID derived from that ID, while the original chunk
    metadata is preserved in the payload.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        self._client = client
        self._collection_name = collection_name
        self._embedding_dimension: int | None = None

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

        await self._ensure_collection(dimension)

        points = [
            models.PointStruct(
                id=self._point_id(item.chunk.id),
                vector=[float(value) for value in item.embedding],
                payload=self._payload(item.chunk),
            )
            for item in chunks
        ]

        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
    ) -> Sequence[RetrievalResult]:
        if top_k <= 0:
            return []

        query = [float(value) for value in embedding]

        if not query:
            return []

        await self._ensure_collection(len(query))

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query,
            limit=top_k,
            with_payload=True,
        )

        results: list[RetrievalResult] = []

        for point in response.points:
            payload = point.payload or {}

            chunk = DocumentChunk(
                id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                content=str(payload["content"]),
                metadata=dict(payload.get("metadata") or {}),
                chunk_index=int(payload["chunk_index"]),
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(point.score),
                )
            )

        return results

    async def _ensure_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")

        if self._embedding_dimension is not None:
            if self._embedding_dimension != dimension:
                raise ValueError(
                    "Embedding dimensions do not match the configured "
                    f"collection dimension: expected "
                    f"{self._embedding_dimension}, got {dimension}."
                )
            return

        exists = await self._client.collection_exists(self._collection_name)

        if exists:
            collection = await self._client.get_collection(self._collection_name)

            configured_dimension = self._collection_dimension(collection)

            if configured_dimension != dimension:
                raise ValueError(
                    "Embedding dimensions do not match the existing "
                    f"Qdrant collection: expected "
                    f"{configured_dimension}, got {dimension}."
                )

            self._embedding_dimension = dimension
            return

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

        self._embedding_dimension = dimension

    @staticmethod
    def _collection_dimension(collection: Any) -> int:
        vectors = collection.config.params.vectors

        if isinstance(vectors, dict):
            if len(vectors) != 1:
                raise ValueError("Qdrant collection must use a single unnamed vector.")
            vectors = next(iter(vectors.values()))

        return int(vectors.size)

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, chunk_id))

    @staticmethod
    def _payload(chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "metadata": dict(chunk.metadata),
            "chunk_index": chunk.chunk_index,
        }
