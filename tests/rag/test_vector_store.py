import pytest

from rag.models import DocumentChunk, EmbeddedChunk
from rag.stores import InMemoryVectorStore


@pytest.mark.asyncio
async def test_vector_store_returns_highest_similarity_first():
    store = InMemoryVectorStore()

    chunk_a = DocumentChunk(
        id="a",
        document_id="doc-1",
        content="Electric vehicles use batteries.",
    )

    chunk_b = DocumentChunk(
        id="b",
        document_id="doc-2",
        content="Gas vehicles use combustion engines.",
    )

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=chunk_a,
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=chunk_b,
                embedding=(0.0, 1.0, 0.0),
            ),
        ]
    )

    results = await store.search(
        embedding=(1.0, 0.0, 0.0),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.id == "a"
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_vector_store_rejects_dimension_mismatch():
    store = InMemoryVectorStore()

    chunk = DocumentChunk(
        id="a",
        document_id="doc-1",
        content="test",
    )

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=chunk,
                embedding=(1.0, 0.0),
            )
        ]
    )

    with pytest.raises(ValueError, match="dimensions"):
        await store.search(
            embedding=(1.0, 0.0, 0.0),
        )
