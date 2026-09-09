import pytest

from rag.models import DocumentChunk, EmbeddedChunk
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore


class FakeEmbeddingService:
    async def embed(self, text: str):
        if "electric" in text.lower():
            return [1.0, 0.0, 0.0]

        return [0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_semantic_retriever_returns_relevant_chunk():
    store = InMemoryVectorStore()

    electric_chunk = DocumentChunk(
        id="electric",
        document_id="vehicle-doc",
        content="Electric vehicles use battery power.",
    )

    gasoline_chunk = DocumentChunk(
        id="gasoline",
        document_id="vehicle-doc",
        content="Gasoline vehicles use combustion engines.",
    )

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=electric_chunk,
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=gasoline_chunk,
                embedding=(0.0, 1.0, 0.0),
            ),
        ]
    )

    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
    )

    results = await retriever.retrieve(
        "How does an electric vehicle work?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "electric"


@pytest.mark.asyncio
async def test_retriever_rejects_empty_query():
    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=InMemoryVectorStore(),
    )

    with pytest.raises(ValueError, match="empty"):
        await retriever.retrieve("")


@pytest.mark.asyncio
async def test_retriever_filters_results_below_min_score():
    store = InMemoryVectorStore()

    relevant_chunk = DocumentChunk(
        id="relevant",
        document_id="vehicle-doc",
        content="Electric vehicles use battery power.",
    )

    unrelated_chunk = DocumentChunk(
        id="unrelated",
        document_id="vehicle-doc",
        content="Gasoline vehicles use combustion engines.",
    )

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=relevant_chunk,
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=unrelated_chunk,
                embedding=(0.0, 1.0, 0.0),
            ),
        ]
    )

    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
    )

    results = await retriever.retrieve(
        "How does an electric vehicle work?",
        top_k=2,
        min_score=0.5,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "relevant"
    assert results[0].score >= 0.5


@pytest.mark.asyncio
async def test_retriever_returns_empty_when_no_result_meets_min_score():
    store = InMemoryVectorStore()

    chunk = DocumentChunk(
        id="gasoline",
        document_id="vehicle-doc",
        content="Gasoline vehicles use combustion engines.",
    )

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=chunk,
                embedding=(0.0, 1.0, 0.0),
            ),
        ]
    )

    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
    )

    results = await retriever.retrieve(
        "How does an electric vehicle work?",
        top_k=5,
        min_score=0.9,
    )

    assert results == []


@pytest.mark.asyncio
async def test_retriever_rejects_invalid_min_score():
    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=InMemoryVectorStore(),
    )

    with pytest.raises(ValueError, match="between -1.0 and 1.0"):
        await retriever.retrieve(
            "electric vehicle",
            min_score=1.1,
        )


@pytest.mark.asyncio
async def test_retriever_filters_by_metadata():
    store = InMemoryVectorStore()

    matching_chunk = DocumentChunk(
        id="matching",
        document_id="doc-1",
        content="Electric vehicle architecture.",
        metadata={"source": "architecture.md"},
    )

    non_matching_chunk = DocumentChunk(
        id="non-matching",
        document_id="doc-2",
        content="Electric vehicle gateway.",
        metadata={"source": "gateway.md"},
    )

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=matching_chunk,
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=non_matching_chunk,
                embedding=(0.9, 0.1, 0.0),
            ),
        ]
    )

    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
    )

    results = await retriever.retrieve(
        "How does an electric vehicle work?",
        top_k=5,
        metadata_filter={"source": "architecture.md"},
    )

    assert len(results) == 1
    assert results[0].chunk.id == "matching"
