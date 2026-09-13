import pytest

from rag.models import DocumentChunk, EmbeddedChunk
from rag.stores.faiss import FAISSVectorStore


@pytest.mark.asyncio
async def test_faiss_vector_store_returns_highest_similarity_first():
    store = FAISSVectorStore()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="a",
                    document_id="doc-1",
                    content="Electric vehicles use batteries.",
                ),
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="b",
                    document_id="doc-2",
                    content="Gas vehicles use combustion engines.",
                ),
                embedding=(0.0, 1.0, 0.0),
            ),
        ]
    )

    results = await store.search(
        embedding=(1.0, 0.0, 0.0),
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["a", "b"]
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_faiss_vector_store_filters_by_metadata():
    store = FAISSVectorStore()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="architecture",
                    document_id="doc-1",
                    content="Architecture document.",
                    metadata={"source": "architecture.md"},
                ),
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="gateway",
                    document_id="doc-2",
                    content="Gateway document.",
                    metadata={"source": "gateway.md"},
                ),
                embedding=(0.9, 0.1, 0.0),
            ),
        ]
    )

    results = await store.search(
        embedding=(1.0, 0.0, 0.0),
        top_k=5,
        metadata_filter={"source": "gateway.md"},
    )

    assert len(results) == 1
    assert results[0].chunk.id == "gateway"


@pytest.mark.asyncio
async def test_faiss_vector_store_rejects_dimension_mismatch():
    store = FAISSVectorStore()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="a",
                    document_id="doc-1",
                    content="test",
                ),
                embedding=(1.0, 0.0),
            )
        ]
    )

    with pytest.raises(ValueError, match="dimensions"):
        await store.search(
            embedding=(1.0, 0.0, 0.0),
        )


@pytest.mark.asyncio
async def test_faiss_vector_store_rejects_mixed_embedding_dimensions():
    store = FAISSVectorStore()

    with pytest.raises(
        ValueError,
        match="All embeddings must have the same dimension",
    ):
        await store.upsert(
            [
                EmbeddedChunk(
                    chunk=DocumentChunk(
                        id="a",
                        document_id="doc-1",
                        content="first",
                    ),
                    embedding=(1.0, 0.0),
                ),
                EmbeddedChunk(
                    chunk=DocumentChunk(
                        id="b",
                        document_id="doc-1",
                        content="second",
                    ),
                    embedding=(1.0, 0.0, 0.0),
                ),
            ]
        )


@pytest.mark.asyncio
async def test_faiss_vector_store_delete_removes_chunks():
    store = FAISSVectorStore()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="a",
                    document_id="doc-1",
                    content="first",
                ),
                embedding=(1.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="b",
                    document_id="doc-1",
                    content="second",
                ),
                embedding=(0.0, 1.0),
            ),
        ]
    )

    await store.delete_chunks(["a"])

    results = await store.search(
        embedding=(1.0, 0.0),
        top_k=5,
    )

    assert [result.chunk.id for result in results] == ["b"]


@pytest.mark.asyncio
async def test_faiss_vector_store_upsert_is_idempotent():
    store = FAISSVectorStore()

    chunk = EmbeddedChunk(
        chunk=DocumentChunk(
            id="a",
            document_id="doc-1",
            content="first",
        ),
        embedding=(1.0, 0.0),
    )

    await store.upsert([chunk])
    await store.upsert([chunk])

    results = await store.search(
        embedding=(1.0, 0.0),
        top_k=5,
    )

    assert [result.chunk.id for result in results] == ["a"]


@pytest.mark.asyncio
async def test_faiss_vector_store_returns_empty_for_empty_query():
    store = FAISSVectorStore()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="a",
                    document_id="doc-1",
                    content="test",
                ),
                embedding=(1.0, 0.0),
            )
        ]
    )

    assert await store.search(embedding=()) == []


@pytest.mark.asyncio
async def test_faiss_vector_store_handles_zero_vectors():
    store = FAISSVectorStore()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="zero",
                    document_id="doc-1",
                    content="zero",
                ),
                embedding=(0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="nonzero",
                    document_id="doc-1",
                    content="nonzero",
                ),
                embedding=(1.0, 0.0),
            ),
        ]
    )

    results = await store.search(
        embedding=(0.0, 1.0),
        top_k=2,
    )

    assert results[0].score == pytest.approx(0.0)
    assert results[1].score == pytest.approx(0.0)
    assert {result.chunk.id for result in results} == {"zero", "nonzero"}
