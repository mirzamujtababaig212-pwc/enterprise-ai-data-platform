from __future__ import annotations
from datetime import UTC, datetime

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
from rag.evaluation.comparison.regression_policy import RetrievalRegressionPolicy
from rag.evaluation.stores.postgres import PostgreSQLRetrievalEvaluationRunStore
from rag.evaluation.models import RetrievalEvaluationResult


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


@pytest.mark.asyncio
async def test_persisted_postgres_baseline_drives_regression_release_decision() -> None:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip(
            "Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL evaluation integration test"
        )

    engine = _postgres_engine()

    baseline_run_id = "vehicle-postgres-regression-baseline"
    candidate_run_id = "vehicle-postgres-regression-candidate"

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

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

        baseline = result.to_run(
            run_id=baseline_run_id,
            created_at=datetime(2026, 9, 13, 10, 0, tzinfo=UTC),
        )

        baseline_session = session_factory()
        try:
            repository = PostgreSQLRetrievalEvaluationRunStore(baseline_session)
            await repository.save(baseline)

            restored_baseline = await repository.get(baseline_run_id)
        finally:
            baseline_session.close()

        assert restored_baseline is not None
        assert restored_baseline.run_id == baseline_run_id
        assert restored_baseline.lineage == baseline.lineage
        assert restored_baseline.evaluation.recall_at_k == baseline.evaluation.recall_at_k
        assert restored_baseline.evaluation.precision_at_k == baseline.evaluation.precision_at_k
        assert restored_baseline.evaluation.mrr == baseline.evaluation.mrr
        assert restored_baseline.evaluation.ndcg_at_k == baseline.evaluation.ndcg_at_k
        assert (
            restored_baseline.evaluation.evaluated_queries == baseline.evaluation.evaluated_queries
        )
        assert (
            restored_baseline.evaluation.successful_queries
            == baseline.evaluation.successful_queries
        )
        assert restored_baseline.evaluation.failed_queries == baseline.evaluation.failed_queries
        assert restored_baseline.evaluation.mean_latency_ms == pytest.approx(
            baseline.evaluation.mean_latency_ms
        )
        assert restored_baseline.evaluation.query_results == ()
        assert restored_baseline.release_passed is True

        degraded_evaluation = RetrievalEvaluationResult(
            recall_at_k=0.9,
            precision_at_k=result.evaluation.precision_at_k,
            mrr=result.evaluation.mrr,
            ndcg_at_k=result.evaluation.ndcg_at_k,
            evaluated_queries=result.evaluation.evaluated_queries,
            successful_queries=result.evaluation.successful_queries,
            failed_queries=result.evaluation.failed_queries,
            mean_latency_ms=result.evaluation.mean_latency_ms,
            query_results=(),
            abstention_accuracy=result.evaluation.abstention_accuracy,
            abstention_evaluated_queries=(result.evaluation.abstention_evaluated_queries),
        )

        degraded_result = result.__class__(
            dataset_name=result.dataset_name,
            evaluation=degraded_evaluation,
            quality_gate=result.quality_gate,
            lineage=result.lineage,
        )

        candidate = degraded_result.to_run(
            run_id=candidate_run_id,
            created_at=datetime(2026, 9, 13, 10, 1, tzinfo=UTC),
            baseline=restored_baseline,
            regression_policy=RetrievalRegressionPolicy(
                name="vehicle-retrieval-regression-v1",
                max_recall_at_k_degradation=0.0,
            ),
        )

        assert candidate.regression is not None
        assert candidate.regression.baseline_run_id == baseline_run_id
        assert candidate.regression.candidate_run_id == candidate_run_id
        assert candidate.regression.result.passed is False
        assert candidate.release_passed is False

        candidate_session = session_factory()
        try:
            repository = PostgreSQLRetrievalEvaluationRunStore(candidate_session)
            await repository.save(candidate)

            restored_candidate = await repository.get(candidate_run_id)
        finally:
            candidate_session.close()

        assert restored_candidate is not None
        assert restored_candidate.regression is not None
        assert restored_candidate.regression.baseline_run_id == baseline_run_id
        assert restored_candidate.regression.candidate_run_id == candidate_run_id
        assert restored_candidate.regression.result.passed is False
        assert restored_candidate.release_passed is False

    finally:
        cleanup_session = session_factory()
        try:
            cleanup_session.execute(
                text(
                    "DELETE FROM retrieval_evaluation_runs "
                    "WHERE run_id IN (:baseline_run_id, :candidate_run_id)"
                ),
                {
                    "baseline_run_id": baseline_run_id,
                    "candidate_run_id": candidate_run_id,
                },
            )
            cleanup_session.commit()
        finally:
            cleanup_session.close()

        _cleanup_benchmark(engine)
        engine.dispose()
