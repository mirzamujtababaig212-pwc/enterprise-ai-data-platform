import pytest

from rag import RAGIndexer
from rag.chunking import RecursiveChunker
from rag.models import Document
from rag.stores import InMemoryVectorStore


class FakeEmbeddingService:
    async def embed(self, text: str):
        checksum = sum(ord(character) for character in text)

        return [
            float(checksum),
            float(len(text)),
        ]


@pytest.mark.asyncio
async def test_indexer_chunks_embeds_and_stores_document():
    chunker = RecursiveChunker(
        chunk_size=5,
        overlap=1,
    )

    embedding_service = FakeEmbeddingService()
    vector_store = InMemoryVectorStore()

    indexer = RAGIndexer(
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    document = Document(
        id="doc-1",
        content="abcdefghij",
        metadata={
            "source": "test",
        },
    )

    embedded_chunks = await indexer.index(document)

    assert len(embedded_chunks) == 3

    assert embedded_chunks[0].chunk.id == "doc-1:chunk:0"
    assert embedded_chunks[1].chunk.id == "doc-1:chunk:1"
    assert embedded_chunks[2].chunk.id == "doc-1:chunk:2"

    assert all(len(item.embedding) == 2 for item in embedded_chunks)

    results = await vector_store.search(
        embedded_chunks[0].embedding,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "doc-1:chunk:0"


@pytest.mark.asyncio
async def test_indexer_preserves_document_metadata():
    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=100,
            overlap=0,
        ),
        embedding_service=FakeEmbeddingService(),
        vector_store=InMemoryVectorStore(),
    )

    document = Document(
        id="doc-metadata",
        content="Enterprise AI",
        metadata={
            "source": "internal",
            "department": "engineering",
        },
    )

    chunks = await indexer.index(document)

    assert len(chunks) == 1

    assert chunks[0].chunk.metadata == {
        "source": "internal",
        "department": "engineering",
    }


@pytest.mark.asyncio
async def test_indexer_handles_empty_document():
    vector_store = InMemoryVectorStore()

    indexer = RAGIndexer(
        chunker=RecursiveChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    document = Document(
        id="empty",
        content="",
    )

    result = await indexer.index(document)

    assert result == []

    results = await vector_store.search(
        embedding=(1.0, 2.0),
        top_k=5,
    )

    assert results == []
