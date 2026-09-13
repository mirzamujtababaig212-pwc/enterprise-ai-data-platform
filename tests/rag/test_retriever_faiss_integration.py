from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingIdentity,
)
from rag.retrieval import SemanticRetriever
from rag.stores.faiss import FAISSVectorStore


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
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id="doc-1",
            content=content,
            metadata=metadata or {},
        ),
        embedding=embedding,
        embedding_identity=_identity(len(embedding)),
    )


@pytest.mark.asyncio
async def test_semantic_retriever_uses_faiss_for_semantic_search() -> None:
    identity = _identity()
    embedding_service = FakeEmbeddingService(
        vectors={
            "architecture query": (1.0, 0.0, 0.0),
        },
        identity=identity,
    )
    store = FAISSVectorStore()

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


@pytest.mark.asyncio
async def test_semantic_retriever_applies_min_score_to_faiss_results() -> None:
    identity = _identity()
    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0),
        },
        identity=identity,
    )
    store = FAISSVectorStore()

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


@pytest.mark.asyncio
async def test_semantic_retriever_passes_metadata_filter_to_faiss() -> None:
    identity = _identity()
    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0),
        },
        identity=identity,
    )
    store = FAISSVectorStore()

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

    assert [result.chunk.id for result in results] == ["gateway"]


@pytest.mark.asyncio
async def test_semantic_retriever_preserves_deterministic_faiss_tie_order() -> None:
    identity = _identity()
    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0),
        },
        identity=identity,
    )
    store = FAISSVectorStore()

    await store.upsert(
        [
            _chunk(
                chunk_id="z-chunk",
                content="Z",
                embedding=(1.0, 0.0, 0.0),
            ),
            _chunk(
                chunk_id="a-chunk",
                content="A",
                embedding=(1.0, 0.0, 0.0),
            ),
        ]
    )

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=store,
    )

    results = await retriever.retrieve(
        "query",
        top_k=2,
    )

    assert [result.chunk.id for result in results] == [
        "a-chunk",
        "z-chunk",
    ]


@pytest.mark.asyncio
async def test_semantic_retriever_validates_faiss_embedding_identity() -> None:
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
    store = FAISSVectorStore()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="incompatible",
                    document_id="doc-1",
                    content="Incompatible embedding",
                ),
                embedding=(1.0, 0.0, 0.0),
                embedding_identity=stored_identity,
            )
        ]
    )

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=store,
    )

    with pytest.raises(ValueError, match="embedding"):
        await retriever.retrieve("query")
