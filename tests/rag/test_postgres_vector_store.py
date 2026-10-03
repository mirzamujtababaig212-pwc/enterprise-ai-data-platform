from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, delete, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.persistence.models import RAGChunkRecord, RAGDocumentRecord
from rag.models import DocumentChunk, EmbeddedChunk, EmbeddingIdentity
from rag.stores.postgres import PostgreSQLVectorStore


def _postgres_store() -> tuple[PostgreSQLVectorStore | None, object | None]:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        return None, None

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        engine.dispose()
        raise

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return PostgreSQLVectorStore(session_factory), engine


def _cleanup(engine: object, document_id: str) -> None:
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session: Session = session_factory()
    try:
        session.execute(delete(RAGChunkRecord).where(RAGChunkRecord.document_id == document_id))
        session.execute(
            delete(RAGDocumentRecord).where(RAGDocumentRecord.document_id == document_id)
        )
        session.commit()
    finally:
        session.close()


def _embedded_chunk(
    *,
    chunk_id: str,
    document_id: str,
    content: str,
    embedding: tuple[float, ...],
    metadata: dict[str, object],
    chunk_index: int,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            content=content,
            metadata=metadata,
            chunk_index=chunk_index,
        ),
        embedding=embedding,
        embedding_identity=EmbeddingIdentity(
            requested_provider="test",
            requested_model="test-embedding",
            resolved_provider="test",
            resolved_model="test-embedding",
            dimension=len(embedding),
        ),
    )


def _ensure_document(engine: object, document_id: str) -> None:
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session: Session = session_factory()
    try:
        session.add(
            RAGDocumentRecord(
                document_id=document_id,
                content="PostgreSQL vector-store integration test",
                document_metadata={"test": True},
            )
        )
        session.commit()
    finally:
        session.close()


def test_postgresql_vector_store_round_trip_preserves_provenance() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    document_id = "postgres-provenance-round-trip"

    source_ref = {
        "platform": "snowflake",
        "object_type": "table",
        "object_name": "ANALYTICS.CUSTOMERS",
        "namespace": "ANALYTICS",
        "environment": "prod",
    }
    locator = {
        "type": "document_section",
        "value": "customer_overview",
    }

    chunks = [
        _embedded_chunk(
            chunk_id=f"{document_id}:chunk:0",
            document_id=document_id,
            content="Customer data is stored in the analytics platform.",
            embedding=(1.0, 0.0, 0.0, 0.0),
            metadata={
                "source": "test",
                "source_ref": source_ref,
                "locator": locator,
            },
            chunk_index=0,
        )
    ]

    try:
        assert inspect(engine).has_table("rag_chunks")
        assert inspect(engine).has_table("rag_documents")

        _ensure_document(engine, document_id)

        import asyncio

        asyncio.run(store.upsert(chunks))

        results = asyncio.run(
            store.search(
                embedding=(1.0, 0.0, 0.0, 0.0),
                top_k=1,
            )
        )

        assert len(results) == 1
        assert results[0].chunk.metadata["source_ref"] == source_ref
        assert results[0].chunk.metadata["locator"] == locator
    finally:
        _cleanup(engine, document_id)
        engine.dispose()


def test_postgresql_vector_store_upsert_search_and_delete() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    document_id = "postgres-vector-store-integration"
    chunks = [
        _embedded_chunk(
            chunk_id=f"{document_id}:chunk:0",
            document_id=document_id,
            content="Enterprise AI platform architecture",
            embedding=(1.0, 0.0, 0.0, 0.0),
            metadata={"category": "architecture"},
            chunk_index=0,
        ),
        _embedded_chunk(
            chunk_id=f"{document_id}:chunk:1",
            document_id=document_id,
            content="RAG retrieval and vector search",
            embedding=(0.8, 0.6, 0.0, 0.0),
            metadata={"category": "rag"},
            chunk_index=1,
        ),
        _embedded_chunk(
            chunk_id=f"{document_id}:chunk:2",
            document_id=document_id,
            content="Unrelated operational content",
            embedding=(0.0, 0.0, 1.0, 0.0),
            metadata={"category": "operations"},
            chunk_index=2,
        ),
    ]

    try:
        assert inspect(engine).has_table("rag_chunks")
        assert inspect(engine).has_table("rag_documents")

        _ensure_document(engine, document_id)

        import asyncio

        asyncio.run(store.upsert(chunks))

        results = asyncio.run(
            store.search(
                embedding=(1.0, 0.0, 0.0, 0.0),
                top_k=3,
            )
        )

        assert [result.chunk.id for result in results] == [
            chunks[0].chunk.id,
            chunks[1].chunk.id,
            chunks[2].chunk.id,
        ]
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.8)
        assert results[2].score == pytest.approx(0.0)

        assert results[0].embedding_identity is not None
        assert results[0].embedding_identity.resolved_provider == "test"
        assert results[0].embedding_identity.resolved_model == "test-embedding"
        assert results[0].embedding_identity.dimension == 4

        filtered = asyncio.run(
            store.search(
                embedding=(1.0, 0.0, 0.0, 0.0),
                top_k=5,
                metadata_filter={"category": "rag"},
            )
        )

        assert [result.chunk.id for result in filtered] == [chunks[1].chunk.id]

        asyncio.run(store.delete_chunks([chunks[1].chunk.id]))

        remaining = asyncio.run(
            store.search(
                embedding=(1.0, 0.0, 0.0, 0.0),
                top_k=5,
            )
        )

        assert [result.chunk.id for result in remaining] == [
            chunks[0].chunk.id,
            chunks[2].chunk.id,
        ]
    finally:
        _cleanup(engine, document_id)
        engine.dispose()


def test_postgresql_vector_store_rejects_mixed_dimensions() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    try:
        first = _embedded_chunk(
            chunk_id="postgres-vector-dimension-0",
            document_id="postgres-vector-dimension-test",
            content="First",
            embedding=(1.0, 0.0, 0.0, 0.0),
            metadata={},
            chunk_index=0,
        )
        second = _embedded_chunk(
            chunk_id="postgres-vector-dimension-1",
            document_id="postgres-vector-dimension-test",
            content="Second",
            embedding=(1.0, 0.0, 0.0),
            metadata={},
            chunk_index=1,
        )

        import asyncio

        with pytest.raises(
            ValueError,
            match="All embeddings must have the same dimension",
        ):
            asyncio.run(store.upsert([first, second]))
    finally:
        engine.dispose()


def test_postgresql_vector_store_rejects_query_dimension_mismatch() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    document_id = "postgres-vector-query-dimension"
    chunk = _embedded_chunk(
        chunk_id=f"{document_id}:chunk:0",
        document_id=document_id,
        content="Dimension validation",
        embedding=(1.0, 0.0, 0.0, 0.0),
        metadata={},
        chunk_index=0,
    )

    try:
        _ensure_document(engine, document_id)

        import asyncio

        asyncio.run(store.upsert([chunk]))

        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            asyncio.run(
                store.search(
                    embedding=(1.0, 0.0, 0.0),
                    top_k=5,
                )
            )
    finally:
        _cleanup(engine, document_id)
        engine.dispose()


def test_postgresql_vector_store_returns_empty_for_empty_query_embedding() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    document_id = "postgres-vector-empty-query"
    chunk = _embedded_chunk(
        chunk_id=f"{document_id}:chunk:0",
        document_id=document_id,
        content="Empty query validation",
        embedding=(1.0, 0.0, 0.0, 0.0),
        metadata={},
        chunk_index=0,
    )

    try:
        _ensure_document(engine, document_id)

        import asyncio

        asyncio.run(store.upsert([chunk]))

        assert asyncio.run(store.search(embedding=())) == []
    finally:
        _cleanup(engine, document_id)
        engine.dispose()


def test_postgresql_vector_store_upsert_replaces_existing_chunk() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    document_id = "postgres-vector-upsert-replace"
    chunk_id = f"{document_id}:chunk:0"

    original = _embedded_chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content="Original content",
        embedding=(1.0, 0.0, 0.0, 0.0),
        metadata={"version": 1},
        chunk_index=0,
    )

    replacement = _embedded_chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content="Replacement content",
        embedding=(0.0, 1.0, 0.0, 0.0),
        metadata={"version": 2},
        chunk_index=3,
    )

    try:
        _ensure_document(engine, document_id)

        import asyncio

        asyncio.run(store.upsert([original]))
        asyncio.run(store.upsert([replacement]))

        results = asyncio.run(
            store.search(
                embedding=(0.0, 1.0, 0.0, 0.0),
                top_k=5,
            )
        )

        assert len(results) == 1
        assert results[0].chunk.id == chunk_id
        assert results[0].chunk.content == "Replacement content"
        assert results[0].chunk.metadata == {"version": 2}
        assert results[0].chunk.chunk_index == 3
        assert results[0].score == pytest.approx(1.0)
    finally:
        _cleanup(engine, document_id)
        engine.dispose()
