import pytest
from qdrant_client import AsyncQdrantClient

from rag.models import DocumentChunk, EmbeddedChunk
from rag.stores.qdrant import QdrantVectorStore


def make_chunk(
    chunk_id: str,
    content: str,
    chunk_index: int,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="document-1",
        content=content,
        metadata={"source": "test"},
        chunk_index=chunk_index,
    )


@pytest.mark.asyncio
async def test_upsert_and_search_round_trip() -> None:
    client = AsyncQdrantClient(location=":memory:")

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name="test_rag",
        )

        chunks = [
            EmbeddedChunk(
                chunk=make_chunk("chunk-1", "alpha", 0),
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=make_chunk("chunk-2", "beta", 1),
                embedding=(0.0, 1.0, 0.0),
            ),
        ]

        await store.upsert(chunks)

        results = await store.search(
            embedding=(1.0, 0.0, 0.0),
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].chunk.id == "chunk-1"
        assert results[0].chunk.document_id == "document-1"
        assert results[0].chunk.content == "alpha"
        assert results[0].chunk.metadata == {"source": "test"}
        assert results[0].chunk.chunk_index == 0
        assert results[0].score > results[1].score
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upsert_is_idempotent_for_same_chunk_id() -> None:
    client = AsyncQdrantClient(location=":memory:")

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name="test_idempotent",
        )

        chunk = make_chunk("chunk-1", "original", 0)

        await store.upsert(
            [
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=(1.0, 0.0),
                )
            ]
        )

        await store.upsert(
            [
                EmbeddedChunk(
                    chunk=make_chunk("chunk-1", "updated", 0),
                    embedding=(0.0, 1.0),
                )
            ]
        )

        results = await store.search(
            embedding=(0.0, 1.0),
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].chunk.id == "chunk-1"
        assert results[0].chunk.content == "updated"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_search_returns_empty_for_non_positive_top_k() -> None:
    client = AsyncQdrantClient(location=":memory:")

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name="test_empty",
        )

        assert await store.search((1.0, 0.0), top_k=0) == []
        assert await store.search((1.0, 0.0), top_k=-1) == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mismatched_embedding_dimensions_are_rejected() -> None:
    client = AsyncQdrantClient(location=":memory:")

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name="test_dimensions",
        )

        await store.upsert(
            [
                EmbeddedChunk(
                    chunk=make_chunk("chunk-1", "alpha", 0),
                    embedding=(1.0, 0.0),
                )
            ]
        )

        with pytest.raises(ValueError, match="dimensions"):
            await store.upsert(
                [
                    EmbeddedChunk(
                        chunk=make_chunk("chunk-2", "beta", 1),
                        embedding=(1.0, 0.0, 0.0),
                    )
                ]
            )
    finally:
        await client.close()
