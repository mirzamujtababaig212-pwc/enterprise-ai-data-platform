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
