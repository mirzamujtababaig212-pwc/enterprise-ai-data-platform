from __future__ import annotations

from dataclasses import dataclass

import pytest
from qdrant_client import AsyncQdrantClient

from rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingIdentity,
)
from rag.retrieval import SemanticRetriever
from rag.stores.qdrant import QdrantVectorStore


@dataclass
class FakeEmbeddingService:
    vectors: dict[str, tuple[float, ...]]
    identity: EmbeddingIdentity

    async def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]

    async def embed_with_metadata(self, text: str):
        from rag.models import EmbeddingResult

        return EmbeddingResult(
            vector=self.vectors[text],
            identity=self.identity,
        )


def _identity(dimension: int = 3) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        requested_provider="test",
        requested_model="test-embedding",
        resolved_provider="test",
        resolved_model="test-embedding",
        dimension=dimension,
    )


def _chunk(
    *,
    chunk_id: str,
    content: str,
    embedding: tuple[float, ...],
    metadata: dict[str, object] | None = None,
    identity: EmbeddingIdentity | None = None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id="doc-1",
            content=content,
            metadata=metadata or {},
        ),
        embedding=embedding,
        embedding_identity=identity or _identity(len(embedding)),
    )


async def _create_store(
    client: AsyncQdrantClient,
    collection_name: str,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )


@pytest.mark.asyncio
async def test_semantic_retriever_uses_qdrant_for_semantic_search() -> None:
    identity = _identity()

    embedding_service = FakeEmbeddingService(
        vectors={
            "architecture query": (1.0, 0.0, 0.0),
        },
        identity=identity,
    )

    client = AsyncQdrantClient(location=":memory:")

    try:
        store = await _create_store(
            client,
            "retriever_qdrant_semantic_search",
        )

        await store.upsert(
            [
                _chunk(
                    chunk_id="architecture",
                    content="Enterprise AI architecture",
                    embedding=(1.0, 0.0, 0.0),
                ),
                _chunk(
                    chunk_id="gateway",
                    content="LLM gateway",
                    embedding=(0.8, 0.6, 0.0),
                ),
                _chunk(
                    chunk_id="operations",
                    content="Operational procedures",
                    embedding=(0.0, 0.0, 1.0),
                ),
            ]
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        results = await retriever.retrieve(
            "architecture query",
            top_k=3,
        )

        assert [result.chunk.id for result in results] == [
            "architecture",
            "gateway",
            "operations",
        ]

        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.8)
        assert results[2].score == pytest.approx(0.0)

    finally:
        await client.close()


@pytest.mark.asyncio
async def test_semantic_retriever_applies_min_score_to_qdrant_results() -> None:
    identity = _identity()

    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0),
        },
        identity=identity,
    )

    client = AsyncQdrantClient(location=":memory:")

    try:
        store = await _create_store(
            client,
            "retriever_qdrant_min_score",
        )

        await store.upsert(
            [
                _chunk(
                    chunk_id="high",
                    content="High relevance",
                    embedding=(1.0, 0.0, 0.0),
                ),
                _chunk(
                    chunk_id="medium",
                    content="Medium relevance",
                    embedding=(0.8, 0.6, 0.0),
                ),
                _chunk(
                    chunk_id="low",
                    content="Low relevance",
                    embedding=(0.0, 0.0, 1.0),
                ),
            ]
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        results = await retriever.retrieve(
            "query",
            top_k=3,
            min_score=0.7,
        )

        assert [result.chunk.id for result in results] == [
            "high",
            "medium",
        ]
        assert all(result.score >= 0.7 for result in results)

    finally:
        await client.close()


@pytest.mark.asyncio
async def test_semantic_retriever_passes_metadata_filter_to_qdrant() -> None:
    identity = _identity()

    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0),
        },
        identity=identity,
    )

    client = AsyncQdrantClient(location=":memory:")

    try:
        store = await _create_store(
            client,
            "retriever_qdrant_metadata_filter",
        )

        await store.upsert(
            [
                _chunk(
                    chunk_id="architecture",
                    content="Architecture",
                    embedding=(1.0, 0.0, 0.0),
                    metadata={"source": "architecture.md"},
                ),
                _chunk(
                    chunk_id="gateway",
                    content="Gateway",
                    embedding=(0.9, 0.1, 0.0),
                    metadata={"source": "gateway.md"},
                ),
            ]
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        results = await retriever.retrieve(
            "query",
            top_k=5,
            metadata_filter={"source": "gateway.md"},
        )

        assert [result.chunk.id for result in results] == [
            "gateway",
        ]

    finally:
        await client.close()


@pytest.mark.asyncio
async def test_semantic_retriever_validates_qdrant_embedding_identity() -> None:
    query_identity = _identity(3)

    stored_identity = EmbeddingIdentity(
        requested_provider="test",
        requested_model="different-model",
        resolved_provider="test",
        resolved_model="different-model",
        dimension=3,
    )

    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0),
        },
        identity=query_identity,
    )

    client = AsyncQdrantClient(location=":memory:")

    try:
        store = await _create_store(
            client,
            "retriever_qdrant_identity",
        )

        await store.upsert(
            [
                _chunk(
                    chunk_id="incompatible",
                    content="Incompatible embedding",
                    embedding=(1.0, 0.0, 0.0),
                    identity=stored_identity,
                )
            ]
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        with pytest.raises(ValueError, match="embedding"):
            await retriever.retrieve("query")

    finally:
        await client.close()
