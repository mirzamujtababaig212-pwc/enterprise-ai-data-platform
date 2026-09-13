from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session, sessionmaker
from qdrant_client import AsyncQdrantClient

from app.control_plane.persistence.models import RAGChunkRecord, RAGDocumentRecord
from rag.evaluation.datasets.vehicle import (
    VEHICLE_EMBEDDING_IDENTITY,
    VehicleBenchmarkEmbeddingService,
    vehicle_benchmark_chunks,
)
from rag.stores.faiss import FAISSVectorStore
from rag.stores.in_memory import InMemoryVectorStore
from rag.stores.postgres import PostgreSQLVectorStore
from rag.stores.qdrant import QdrantVectorStore


async def _query_store(store, query: str, *, top_k: int = 3, metadata_filter=None):
    service = VehicleBenchmarkEmbeddingService()
    embedding = await service.embed_with_metadata(query)

    results = await store.search(
        embedding.vector,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    return results


def _assert_results_equivalent(
    expected,
    actual,
    *,
    score_abs: float = 1e-5,
) -> None:
    assert [result.chunk.id for result in actual] == [result.chunk.id for result in expected]

    assert [result.chunk.document_id for result in actual] == [
        result.chunk.document_id for result in expected
    ]

    for expected_result, actual_result in zip(expected, actual):
        assert actual_result.score == pytest.approx(
            expected_result.score,
            abs=score_abs,
        )
        assert actual_result.embedding_identity == VEHICLE_EMBEDDING_IDENTITY


async def _build_in_memory_store():
    store = InMemoryVectorStore()
    await store.upsert(vehicle_benchmark_chunks())
    return store


async def _build_faiss_store():
    store = FAISSVectorStore()
    await store.upsert(vehicle_benchmark_chunks())
    return store


@pytest.mark.asyncio
async def test_in_memory_and_faiss_match_vehicle_retrieval_contract() -> None:
    in_memory = await _build_in_memory_store()
    faiss_store = await _build_faiss_store()

    queries = (
        "How is an electric vehicle powered?",
        "How does regenerative braking work?",
        "How is an electric vehicle battery charged?",
        "How is a vehicle maintained?",
    )

    for query in queries:
        expected = await _query_store(in_memory, query, top_k=3)
        actual = await _query_store(faiss_store, query, top_k=3)

        _assert_results_equivalent(expected, actual)


@pytest.mark.asyncio
async def test_in_memory_and_faiss_match_metadata_filtering() -> None:
    in_memory = await _build_in_memory_store()
    faiss_store = await _build_faiss_store()

    metadata_filter = {"domain": "vehicle"}

    expected = await _query_store(
        in_memory,
        "How is an electric vehicle powered?",
        top_k=5,
        metadata_filter=metadata_filter,
    )
    actual = await _query_store(
        faiss_store,
        "How is an electric vehicle powered?",
        top_k=5,
        metadata_filter=metadata_filter,
    )

    _assert_results_equivalent(expected, actual)


def _postgres_engine():
    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )


def _cleanup_postgres_benchmark(engine) -> None:
    chunks = vehicle_benchmark_chunks()
    chunk_ids = [item.chunk.id for item in chunks]
    document_ids = {item.chunk.document_id for item in chunks}

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session: Session = session_factory()

    try:
        session.execute(delete(RAGChunkRecord).where(RAGChunkRecord.chunk_id.in_(chunk_ids)))
        session.execute(
            delete(RAGDocumentRecord).where(RAGDocumentRecord.document_id.in_(document_ids))
        )
        session.commit()
    finally:
        session.close()


def _ensure_postgres_benchmark_documents(engine) -> None:
    document_ids = {item.chunk.document_id for item in vehicle_benchmark_chunks()}

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session: Session = session_factory()

    try:
        for document_id in document_ids:
            existing = session.scalar(
                text(
                    "SELECT document_id " "FROM rag_documents " "WHERE document_id = :document_id"
                ),
                {"document_id": document_id},
            )

            if existing is None:
                session.add(
                    RAGDocumentRecord(
                        document_id=document_id,
                        content="Vehicle retrieval backend parity benchmark",
                        document_metadata={"evaluation": "vehicle-retrieval"},
                    )
                )

        session.commit()
    finally:
        session.close()


@pytest.mark.asyncio
async def test_postgres_matches_in_memory_vehicle_retrieval_contract() -> None:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL backend parity")

    engine = _postgres_engine()

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        _cleanup_postgres_benchmark(engine)
        _ensure_postgres_benchmark_documents(engine)

        reference = await _build_in_memory_store()

        postgres = PostgreSQLVectorStore()

        await postgres.upsert(vehicle_benchmark_chunks())

        queries = (
            "How is an electric vehicle powered?",
            "How does regenerative braking work?",
            "How is an electric vehicle battery charged?",
            "How is a vehicle maintained?",
        )

        for query in queries:
            expected = await _query_store(reference, query, top_k=3)
            actual = await _query_store(postgres, query, top_k=3)

            _assert_results_equivalent(
                expected,
                actual,
                score_abs=1e-4,
            )

    finally:
        _cleanup_postgres_benchmark(engine)
        engine.dispose()


@pytest.mark.asyncio
async def test_qdrant_matches_in_memory_vehicle_retrieval_contract() -> None:
    if os.getenv("RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip("Set RUN_QDRANT_INTEGRATION=1 to run Qdrant backend parity")

    url = os.getenv("QDRANT_TEST_URL", "http://localhost:6333")
    collection = os.getenv(
        "QDRANT_TEST_COLLECTION",
        "vehicle-retrieval-backend-parity",
    )

    client = AsyncQdrantClient(url=url)

    try:
        assert await client.get_collections() is not None

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        reference = await _build_in_memory_store()

        qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection,
        )

        await qdrant.upsert(vehicle_benchmark_chunks())

        queries = (
            "How is an electric vehicle powered?",
            "How does regenerative braking work?",
            "How is an electric vehicle battery charged?",
            "How is a vehicle maintained?",
        )

        for query in queries:
            expected = await _query_store(reference, query, top_k=3)
            actual = await _query_store(qdrant, query, top_k=3)

            _assert_results_equivalent(
                expected,
                actual,
                score_abs=1e-4,
            )

    finally:
        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        await client.close()
