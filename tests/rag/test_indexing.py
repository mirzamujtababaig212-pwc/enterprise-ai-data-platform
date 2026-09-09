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


@pytest.mark.asyncio
async def test_indexer_removes_stale_chunks_after_successful_reindex():
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

    original = Document(
        id="reindex-doc",
        content="abcdefghij",
    )

    original_chunks = await indexer.index(original)

    assert len(original_chunks) == 3

    updated = Document(
        id="reindex-doc",
        content="abcde",
    )

    updated_chunks = await indexer.index(
        updated,
        previous_chunk_ids=[chunk.chunk.id for chunk in original_chunks],
    )

    assert len(updated_chunks) == 1
    assert updated_chunks[0].chunk.id == "reindex-doc:chunk:0"

    assert set(vector_store._items) == {
        "reindex-doc:chunk:0",
    }


@pytest.mark.asyncio
async def test_indexer_preserves_previous_vectors_when_embedding_fails():
    class FailingEmbeddingService:
        def __init__(self):
            self.calls = 0

        async def embed(self, text: str):
            self.calls += 1

            if self.calls > 1:
                raise RuntimeError("embedding failed")

            return [
                float(sum(ord(character) for character in text)),
                float(len(text)),
            ]

    vector_store = InMemoryVectorStore()

    first_embedding_service = FakeEmbeddingService()

    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=100,
            overlap=0,
        ),
        embedding_service=first_embedding_service,
        vector_store=vector_store,
    )

    original = Document(
        id="failure-doc",
        content="original knowledge",
    )

    original_chunks = await indexer.index(original)

    assert len(original_chunks) == 1

    failing_indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=5,
            overlap=1,
        ),
        embedding_service=FailingEmbeddingService(),
        vector_store=vector_store,
    )

    updated = Document(
        id="failure-doc",
        content="updated knowledge that requires multiple chunks",
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        await failing_indexer.index(
            updated,
            previous_chunk_ids=[chunk.chunk.id for chunk in original_chunks],
        )

    assert set(vector_store._items) == {
        "failure-doc:chunk:0",
    }
