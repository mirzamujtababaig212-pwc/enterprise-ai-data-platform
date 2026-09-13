from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.persistence.models import RAGChunkRecord, RAGDocumentRecord
from rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingIdentity,
    EmbeddingResult,
)
from rag.retrieval import SemanticRetriever
from rag.stores.postgres import PostgreSQLVectorStore


@dataclass
class FakeEmbeddingService:
    vectors: dict[str, tuple[float, ...]]
    identity: EmbeddingIdentity

    async def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=self.vectors[text],
            identity=self.identity,
        )


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
                content="PostgreSQL retriever integration test",
                document_metadata={"test": True},
            )
        )
        session.commit()
    finally:
        session.close()


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


def _identity(dimension: int = 4) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        requested_provider="test",
        requested_model="test-embedding",
        resolved_provider="test",
        resolved_model="test-embedding",
        dimension=dimension,
    )


def _chunk(
    *,
    document_id: str,
    chunk_id: str,
    content: str,
    embedding: tuple[float, ...],
    metadata: dict[str, object] | None = None,
    identity: EmbeddingIdentity | None = None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            content=content,
            metadata=metadata or {},
            chunk_index=0,
        ),
        embedding=embedding,
        embedding_identity=identity or _identity(len(embedding)),
    )


def _require_store() -> tuple[PostgreSQLVectorStore, object]:
    store, engine = _postgres_store()

    if store is None or engine is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    return store, engine


def test_semantic_retriever_uses_postgresql_for_semantic_search() -> None:
    store, engine = _require_store()
    document_id = "retriever-postgres-semantic-search"

    identity = _identity()

    embedding_service = FakeEmbeddingService(
        vectors={
            "architecture query": (1.0, 0.0, 0.0, 0.0),
        },
        identity=identity,
    )

    try:
        _ensure_document(engine, document_id)

        asyncio.run(
            store.upsert(
                [
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:architecture",
                        content="Enterprise AI architecture",
                        embedding=(1.0, 0.0, 0.0, 0.0),
                    ),
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:gateway",
                        content="LLM gateway",
                        embedding=(0.8, 0.6, 0.0, 0.0),
                    ),
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:operations",
                        content="Operational procedures",
                        embedding=(0.0, 0.0, 1.0, 0.0),
                    ),
                ]
            )
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        results = asyncio.run(
            retriever.retrieve(
                "architecture query",
                top_k=3,
            )
        )

        assert [result.chunk.id for result in results] == [
            f"{document_id}:architecture",
            f"{document_id}:gateway",
            f"{document_id}:operations",
        ]
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.8)
        assert results[2].score == pytest.approx(0.0)

    finally:
        _cleanup(engine, document_id)
        engine.dispose()


def test_semantic_retriever_applies_min_score_to_postgresql_results() -> None:
    store, engine = _require_store()
    document_id = "retriever-postgres-min-score"

    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0, 0.0),
        },
        identity=_identity(),
    )

    try:
        _ensure_document(engine, document_id)

        asyncio.run(
            store.upsert(
                [
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:high",
                        content="High relevance",
                        embedding=(1.0, 0.0, 0.0, 0.0),
                    ),
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:medium",
                        content="Medium relevance",
                        embedding=(0.8, 0.6, 0.0, 0.0),
                    ),
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:low",
                        content="Low relevance",
                        embedding=(0.0, 0.0, 1.0, 0.0),
                    ),
                ]
            )
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        results = asyncio.run(
            retriever.retrieve(
                "query",
                top_k=3,
                min_score=0.7,
            )
        )

        assert [result.chunk.id for result in results] == [
            f"{document_id}:high",
            f"{document_id}:medium",
        ]
        assert all(result.score >= 0.7 for result in results)

    finally:
        _cleanup(engine, document_id)
        engine.dispose()


def test_semantic_retriever_passes_metadata_filter_to_postgresql() -> None:
    store, engine = _require_store()
    document_id = "retriever-postgres-metadata-filter"

    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0, 0.0),
        },
        identity=_identity(),
    )

    try:
        _ensure_document(engine, document_id)

        asyncio.run(
            store.upsert(
                [
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:architecture",
                        content="Architecture",
                        embedding=(1.0, 0.0, 0.0, 0.0),
                        metadata={"source": "architecture.md"},
                    ),
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:gateway",
                        content="Gateway",
                        embedding=(0.9, 0.1, 0.0, 0.0),
                        metadata={"source": "gateway.md"},
                    ),
                ]
            )
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        results = asyncio.run(
            retriever.retrieve(
                "query",
                top_k=5,
                metadata_filter={"source": "gateway.md"},
            )
        )

        assert [result.chunk.id for result in results] == [
            f"{document_id}:gateway",
        ]

    finally:
        _cleanup(engine, document_id)
        engine.dispose()


def test_semantic_retriever_validates_postgresql_embedding_identity() -> None:
    store, engine = _require_store()
    document_id = "retriever-postgres-identity"

    query_identity = _identity(4)
    stored_identity = EmbeddingIdentity(
        requested_provider="test",
        requested_model="different-model",
        resolved_provider="test",
        resolved_model="different-model",
        dimension=4,
    )

    embedding_service = FakeEmbeddingService(
        vectors={
            "query": (1.0, 0.0, 0.0, 0.0),
        },
        identity=query_identity,
    )

    try:
        _ensure_document(engine, document_id)

        asyncio.run(
            store.upsert(
                [
                    _chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:incompatible",
                        content="Incompatible embedding",
                        embedding=(1.0, 0.0, 0.0, 0.0),
                        identity=stored_identity,
                    )
                ]
            )
        )

        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        with pytest.raises(ValueError, match="embedding"):
            asyncio.run(retriever.retrieve("query"))

    finally:
        _cleanup(engine, document_id)
        engine.dispose()
