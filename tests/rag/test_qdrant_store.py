import pytest
from qdrant_client import AsyncQdrantClient

from rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingIdentity,
)
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


def make_identity(
    dimension: int,
) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        requested_provider="test-provider",
        requested_model="test-model",
        resolved_provider="test-provider",
        resolved_model="test-model",
        dimension=dimension,
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


@pytest.mark.asyncio
async def test_delete_chunks_removes_selected_chunks() -> None:
    client = AsyncQdrantClient(location=":memory:")

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name="test_delete_chunks",
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

        await store.delete_chunks(["chunk-1"])

        results = await store.search(
            embedding=(0.0, 1.0, 0.0),
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].chunk.id == "chunk-2"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_search_filters_by_metadata():
    client = AsyncQdrantClient(location=":memory:")

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name="test_metadata_filter",
        )

        first = DocumentChunk(
            id="chunk-1",
            document_id="document-1",
            content="architecture",
            metadata={
                "source": "architecture.md",
                "tenant_id": "tenant-a",
            },
            chunk_index=0,
        )

        second = DocumentChunk(
            id="chunk-2",
            document_id="document-2",
            content="gateway",
            metadata={
                "source": "gateway.md",
                "tenant_id": "tenant-a",
            },
            chunk_index=0,
        )

        await store.upsert(
            [
                EmbeddedChunk(
                    chunk=first,
                    embedding=(1.0, 0.0, 0.0),
                ),
                EmbeddedChunk(
                    chunk=second,
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
        assert results[0].chunk.id == "chunk-2"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_search_applies_multiple_metadata_filters():
    client = AsyncQdrantClient(location=":memory:")

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name="test_metadata_multi_filter",
        )

        chunks = [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="match",
                    document_id="document-1",
                    content="matching document",
                    metadata={
                        "source": "architecture.md",
                        "tenant_id": "tenant-a",
                    },
                    chunk_index=0,
                ),
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="wrong-tenant",
                    document_id="document-2",
                    content="wrong tenant",
                    metadata={
                        "source": "architecture.md",
                        "tenant_id": "tenant-b",
                    },
                    chunk_index=0,
                ),
                embedding=(0.99, 0.01, 0.0),
            ),
        ]

        await store.upsert(chunks)

        results = await store.search(
            embedding=(1.0, 0.0, 0.0),
            top_k=5,
            metadata_filter={
                "source": "architecture.md",
                "tenant_id": "tenant-a",
            },
        )

        assert len(results) == 1
        assert results[0].chunk.id == "match"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_search_preserves_embedding_identity():
    client = AsyncQdrantClient(location=":memory:")

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name="test_embedding_identity",
        )

        identity = make_identity(3)

        await store.upsert(
            [
                EmbeddedChunk(
                    chunk=make_chunk(
                        "chunk-1",
                        "alpha",
                        0,
                    ),
                    embedding=(1.0, 0.0, 0.0),
                    embedding_identity=identity,
                )
            ]
        )

        results = await store.search(
            embedding=(1.0, 0.0, 0.0),
            top_k=5,
        )

        assert len(results) == 1

        assert results[0].embedding_identity == identity

    finally:
        await client.close()
