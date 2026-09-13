from __future__ import annotations

import os

import pytest
from sqlalchemy import delete, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.persistence.models import RAGChunkRecord, RAGDocumentRecord
from rag.evaluation.dataset_registry import VehicleRetrievalEvaluationDatasetDefinition
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.evaluation.datasets.vehicle import vehicle_benchmark_chunks


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


def _cleanup_benchmark(engine) -> None:
    benchmark_chunks = vehicle_benchmark_chunks()
    document_ids = {item.chunk.document_id for item in benchmark_chunks}
    chunk_ids = [item.chunk.id for item in benchmark_chunks]

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


def _ensure_benchmark_documents(engine) -> None:
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
                        content="Vehicle retrieval evaluation benchmark",
                        document_metadata={"evaluation": "vehicle-retrieval"},
                    )
                )

        session.commit()
    finally:
        session.close()


@pytest.mark.asyncio
async def test_vehicle_benchmark_workflow_preserves_postgres_retrieval_lineage() -> None:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip(
            "Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL evaluation integration test"
        )

    engine = _postgres_engine()

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        _cleanup_benchmark(engine)
        _ensure_benchmark_documents(engine)

        definition = VehicleRetrievalEvaluationDatasetDefinition(
            vector_store_backend="postgres",
        )

        retriever = await definition.build_retriever()

        evaluator = RetrievalEvaluator(
            retriever,
            k=3,
            embedding_identity=definition.build_embedding_identity(),
        )

        policy = RetrievalEvaluationPolicy(
            name="vehicle-retrieval-quality-v2",
            min_recall_at_k=1.0,
            min_precision_at_k=0.8,
            min_mrr=1.0,
            min_ndcg_at_k=0.95,
        )

        workflow = RetrievalEvaluationWorkflow(
            evaluator=evaluator,
            policy=policy,
            retrieval_artifact=definition.build_retrieval_artifact(),
        )

        result = await workflow.run(definition.build_dataset())

        assert result.passed is True
        assert result.evaluation.evaluated_queries == 7
        assert result.evaluation.successful_queries == 7
        assert result.evaluation.failed_queries == 0

        assert result.evaluation.recall_at_k == pytest.approx(1.0)
        assert result.evaluation.precision_at_k == pytest.approx(0.8571428571)
        assert result.evaluation.mrr == pytest.approx(1.0)
        assert result.evaluation.ndcg_at_k == pytest.approx(0.9775, abs=1e-4)

        assert result.lineage.retrieval_artifact is not None
        assert result.lineage.retrieval_artifact.retriever_type == "SemanticRetriever"
        assert result.lineage.retrieval_artifact.vector_store_type == "PostgreSQLVectorStore"

        lineage = result.as_dict()["lineage"]

        assert lineage["retriever_type"] == "SemanticRetriever"
        assert lineage["vector_store_type"] == "PostgreSQLVectorStore"

    finally:
        _cleanup_benchmark(engine)
        engine.dispose()
