from __future__ import annotations

import asyncio
import os

import pytest

from datetime import UTC, datetime

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import Base, RetrievalEvaluationRunRecord
from rag.evaluation.composite_release import CompositeEvaluationReleaseDecision
from rag.evaluation.external import (
    ExternalEvaluationMetricPolicy,
    ExternalEvaluationPolicy,
    ExternalEvaluationResult,
)
from rag.evaluation.external.release import (
    ExternalEvaluationReleaseGate,
    ExternalEvaluationReleasePolicy,
)
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.stores.release_decision import (
    PostgreSQLRetrievalEvaluationReleaseDecisionStore,
)
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGate
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.stores.postgres import (
    PostgreSQLRetrievalEvaluationRunStore,
)
from rag.models import EmbeddingIdentity

EMBEDDING_IDENTITY = EmbeddingIdentity(
    requested_provider="logical-provider",
    requested_model="logical-model",
    resolved_provider="physical-provider",
    resolved_model="physical-model",
    dimension=4,
)


def _run(
    *,
    run_id: str = "run-1",
    recall: float = 1.0,
    precision: float = 0.9,
    passed: bool = True,
) -> RetrievalEvaluationRun:
    from rag.evaluation.models import RetrievalEvaluationResult
    from rag.evaluation.lineage import RetrievalEvaluationLineage

    policy = RetrievalEvaluationPolicy(
        min_recall_at_k=0.8,
        min_precision_at_k=0.8,
        name="vehicle-release",
    )

    evaluation = RetrievalEvaluationResult(
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=1.0,
        ndcg_at_k=0.95,
        evaluated_queries=7,
        successful_queries=7,
        failed_queries=0,
        mean_latency_ms=12.5,
        query_results=(),
        abstention_accuracy=1.0,
        abstention_evaluated_queries=1,
    )

    quality_gate = RetrievalQualityGate.evaluate(
        evaluation,
        policy,
    )

    if not passed:
        from rag.evaluation.quality_gate import RetrievalQualityGateResult

        quality_gate = RetrievalQualityGateResult(
            passed=False,
            errors=("forced failure",),
            metrics=quality_gate.metrics,
            policy=policy,
        )

    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        evaluation_policy_name=policy.name,
        min_recall_at_k=policy.min_recall_at_k,
        min_precision_at_k=policy.min_precision_at_k,
        min_mrr=policy.min_mrr,
        min_ndcg_at_k=policy.min_ndcg_at_k,
        max_mean_latency_ms=policy.max_mean_latency_ms,
        min_abstention_accuracy=policy.min_abstention_accuracy,
        evaluator_k=3,
        min_relevance_score=0.2,
        embedding_identity=EMBEDDING_IDENTITY,
    )

    run = RetrievalEvaluationRun(
        run_id=run_id,
        created_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        lineage=lineage,
        evaluation=evaluation,
        quality_gate=quality_gate,
        external_evaluations=(
            ExternalEvaluationResult(
                provider="ragas",
                evaluator="faithfulness",
                metrics={"faithfulness": 0.91},
                evaluated_samples=7,
                metadata={},
            ),
        ),
    )

    external_policy = ExternalEvaluationPolicy(
        name="ragas-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
        ),
    )

    return run.with_external_quality_gate(external_policy)


def _hybrid_run(run_id: str = "hybrid-run-1") -> RetrievalEvaluationRun:
    from dataclasses import replace

    from rag.evaluation.lineage import (
        HybridRetrievalConfiguration,
        RetrievalEvaluationArtifact,
    )

    run = _run(run_id=run_id)

    hybrid_artifact = RetrievalEvaluationArtifact(
        retriever_type="HybridRetriever",
        vector_store_type="InMemoryVectorStore",
        hybrid_configuration=HybridRetrievalConfiguration(
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        ),
    )

    return replace(
        run,
        lineage=replace(
            run.lineage,
            retrieval_artifact=hybrid_artifact,
        ),
    )


def _repository():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return (
        PostgreSQLRetrievalEvaluationRunStore(session_factory()),
        engine,
    )


def test_schema_contains_evaluation_run_table() -> None:
    repository, engine = _repository()

    try:
        assert inspect(engine).has_table("retrieval_evaluation_runs")
    finally:
        repository._session.close()
        engine.dispose()


def test_save_and_get_round_trip_preserves_release_evidence() -> None:
    repository, engine = _repository()

    try:
        run = _run()

        asyncio.run(repository.save(run))
        restored = asyncio.run(repository.get(run.run_id))

        assert restored is not None
        assert restored.run_id == run.run_id
        assert restored.created_at == run.created_at
        assert restored.release_passed == run.release_passed

        assert restored.lineage == run.lineage

        assert restored.evaluation.recall_at_k == run.evaluation.recall_at_k
        assert restored.evaluation.precision_at_k == run.evaluation.precision_at_k
        assert restored.evaluation.mrr == run.evaluation.mrr
        assert restored.evaluation.ndcg_at_k == run.evaluation.ndcg_at_k
        assert restored.evaluation.evaluated_queries == run.evaluation.evaluated_queries
        assert restored.evaluation.mean_latency_ms == run.evaluation.mean_latency_ms
        assert restored.evaluation.query_results == ()

        assert restored.quality_gate.passed == run.quality_gate.passed
        assert restored.quality_gate.errors == run.quality_gate.errors
        assert restored.quality_gate.policy == run.quality_gate.policy
        assert restored.external_evaluations == run.external_evaluations

        assert restored.external_quality_gate is not None
        assert restored.external_quality_gate.passed is True
        assert restored.external_quality_gate.errors == ()
        assert restored.external_quality_gate.metrics == {
            "ragas/faithfulness/faithfulness": 0.91,
        }
        assert restored.external_quality_gate.policy == (run.external_quality_gate.policy)
    finally:
        repository._session.close()
        engine.dispose()


def test_save_and_get_round_trip_preserves_hybrid_retrieval_configuration() -> None:
    repository, engine = _repository()

    try:
        run = _hybrid_run()

        asyncio.run(repository.save(run))
        restored = asyncio.run(repository.get(run.run_id))

        assert restored is not None
        assert restored.lineage == run.lineage

        artifact = restored.lineage.retrieval_artifact
        assert artifact is not None
        assert artifact.retriever_type == "HybridRetriever"
        assert artifact.vector_store_type == "InMemoryVectorStore"

        configuration = artifact.hybrid_configuration
        assert configuration is not None
        assert configuration.candidate_k == 5
        assert configuration.rrf_k == 60
        assert configuration.semantic_weight == 1.0
        assert configuration.lexical_weight == 0.5
    finally:
        repository._session.close()
        engine.dispose()


def test_persisted_run_and_composite_decision_remain_auditable() -> None:
    run_repository, engine = _repository()
    decision_repository = PostgreSQLRetrievalEvaluationReleaseDecisionStore(run_repository._session)

    try:
        original = _run()

        # Native retrieval evaluation passes, but the final release policy
        # requires external evidence that is intentionally absent.
        run = RetrievalEvaluationRun(
            run_id="auditability-run",
            created_at=original.created_at,
            lineage=original.lineage,
            evaluation=original.evaluation,
            quality_gate=original.quality_gate,
            regression=original.regression,
            external_evaluations=(),
            external_quality_gate=None,
        )

        external_release_policy = ExternalEvaluationReleasePolicy(
            name="application-external-evaluation-release",
            required=True,
        )

        native_decision = RetrievalEvaluationReleaseGate.evaluate(run)
        external_decision = ExternalEvaluationReleaseGate.evaluate(
            quality_gate=run.external_quality_gate,
            policy=external_release_policy,
        )

        decision = CompositeEvaluationReleaseDecision(
            run_id=run.run_id,
            native=native_decision,
            external=external_decision,
            passed=native_decision.passed and external_decision.passed,
            errors=tuple(
                [f"native: {error}" for error in native_decision.errors]
                + [f"external: {error}" for error in external_decision.errors]
            ),
        )

        asyncio.run(run_repository.save(run))
        asyncio.run(decision_repository.save(decision))

        restored_run = asyncio.run(run_repository.get(run.run_id))
        restored_decision = asyncio.run(decision_repository.get(run.run_id))

        assert restored_run is not None
        assert restored_decision is not None

        # The native evaluation remains independently successful.
        assert restored_run.release_passed is True
        assert restored_run.external_evaluations == ()
        assert restored_run.external_quality_gate is None

        # The composite release decision remains independently explainable.
        assert restored_decision.native.passed is True
        assert restored_decision.external.passed is False
        assert restored_decision.external.policy.name == ("application-external-evaluation-release")
        assert restored_decision.external.policy.required is True
        assert restored_decision.external.errors == (
            "required external evaluation evidence is missing",
        )

        assert restored_decision.passed is False
        assert restored_decision.errors == (
            "external: required external evaluation evidence is missing",
        )

        # The persisted artifacts together explain the final release outcome
        # without rerunning retrieval evaluation.
        assert restored_decision.run_id == restored_run.run_id
        assert restored_decision.native.run_id == restored_run.run_id
    finally:
        run_repository._session.close()
        engine.dispose()


def test_save_and_get_without_external_quality_gate_preserves_none() -> None:
    repository, engine = _repository()

    try:
        run = _run().with_external_quality_gate(
            ExternalEvaluationPolicy(
                name="ragas-v1",
                metrics=(
                    ExternalEvaluationMetricPolicy(
                        provider="ragas",
                        evaluator="faithfulness",
                        metric_name="faithfulness",
                        minimum_value=0.90,
                    ),
                ),
            )
        )

        run_without_gate = RetrievalEvaluationRun(
            run_id="run-without-external-gate",
            created_at=run.created_at,
            lineage=run.lineage,
            evaluation=run.evaluation,
            quality_gate=run.quality_gate,
            external_evaluations=run.external_evaluations,
        )

        asyncio.run(repository.save(run_without_gate))
        restored = asyncio.run(repository.get(run_without_gate.run_id))

        assert restored is not None
        assert restored.external_quality_gate is None
    finally:
        repository._session.close()
        engine.dispose()


def test_get_missing_run_returns_none() -> None:
    repository, engine = _repository()

    try:
        assert asyncio.run(repository.get("missing")) is None
    finally:
        repository._session.close()
        engine.dispose()


def test_save_rejects_duplicate_run_without_overwriting_existing_run() -> None:
    repository, engine = _repository()

    try:
        from rag.evaluation.run_store import DuplicateEvaluationRunError

        first = _run(recall=0.8)
        second = _run(recall=1.0)

        asyncio.run(repository.save(first))

        with pytest.raises(
            DuplicateEvaluationRunError,
            match="evaluation run already exists: run-1",
        ):
            asyncio.run(repository.save(second))

        restored = asyncio.run(repository.get(first.run_id))

        assert restored is not None
        assert restored.evaluation.recall_at_k == 0.8
    finally:
        repository._session.close()
        engine.dispose()


def test_persisted_evaluation_excludes_query_payload() -> None:
    repository, engine = _repository()

    try:
        run = _run()
        asyncio.run(repository.save(run))

        record = repository._session.get(
            RetrievalEvaluationRunRecord,
            run.run_id,
        )

        assert record is not None

        assert "query" not in record.evaluation
        assert "query_results" not in record.evaluation
        assert "retrieved_chunk_ids" not in record.evaluation
        assert "retrieved_results" not in record.evaluation
        assert "error_message" not in record.evaluation

        assert record.external_evaluations == [
            {
                "provider": "ragas",
                "evaluator": "faithfulness",
                "metrics": {"faithfulness": 0.91},
                "evaluated_samples": 7,
                "metadata": {},
            }
        ]

        assert record.external_quality_gate == {
            "quality_gate_passed": True,
            "quality_gate_policy": "ragas-v1",
            "quality_gate_policy_data": {
                "name": "ragas-v1",
                "metrics": [
                    {
                        "provider": "ragas",
                        "evaluator": "faithfulness",
                        "metric_name": "faithfulness",
                        "minimum_value": 0.90,
                        "maximum_value": None,
                        "required": True,
                    }
                ],
            },
            "quality_gate_errors": [],
            "quality_gate_metrics": {
                "ragas/faithfulness/faithfulness": 0.91,
            },
        }
    finally:
        repository._session.close()
        engine.dispose()


def _postgres_repository():
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
    except OperationalError:
        engine.dispose()
        raise

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return PostgreSQLRetrievalEvaluationRunStore(session_factory()), engine


def test_postgresql_hybrid_retrieval_configuration_round_trip() -> None:
    repository, engine = _postgres_repository()

    if repository is None:
        import pytest

        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    run_id = "postgres-integration-hybrid-evaluation-run"

    try:
        assert inspect(engine).has_table("retrieval_evaluation_runs")

        run = _hybrid_run(run_id=run_id)

        asyncio.run(repository.save(run))
        restored = asyncio.run(repository.get(run_id))

        assert restored is not None
        assert restored.lineage == run.lineage

        artifact = restored.lineage.retrieval_artifact
        assert artifact is not None
        assert artifact.retriever_type == "HybridRetriever"
        assert artifact.vector_store_type == "InMemoryVectorStore"

        configuration = artifact.hybrid_configuration
        assert configuration is not None
        assert configuration.candidate_k == 5
        assert configuration.rrf_k == 60
        assert configuration.semantic_weight == 1.0
        assert configuration.lexical_weight == 0.5
    finally:
        if repository is not None:
            repository._session.execute(
                text("DELETE FROM retrieval_evaluation_runs " "WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            repository._session.commit()
            repository._session.close()

        if engine is not None:
            engine.dispose()


def test_postgresql_save_and_get_round_trip() -> None:
    repository, engine = _postgres_repository()

    if repository is None:
        import pytest

        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    run_id = "postgres-integration-evaluation-run"

    try:
        assert inspect(engine).has_table("retrieval_evaluation_runs")

        run = _run(run_id=run_id)

        asyncio.run(repository.save(run))
        restored = asyncio.run(repository.get(run_id))

        assert restored is not None
        assert restored.run_id == run.run_id

        # PostgreSQL preserves timezone information for timestamptz.
        assert restored.created_at.tzinfo is not None
        assert restored.created_at == run.created_at

        assert restored.release_passed == run.release_passed
        assert restored.lineage == run.lineage

        assert restored.evaluation.recall_at_k == run.evaluation.recall_at_k
        assert restored.evaluation.precision_at_k == run.evaluation.precision_at_k
        assert restored.evaluation.mrr == run.evaluation.mrr
        assert restored.evaluation.ndcg_at_k == run.evaluation.ndcg_at_k
        assert restored.evaluation.evaluated_queries == run.evaluation.evaluated_queries
        assert restored.evaluation.successful_queries == run.evaluation.successful_queries
        assert restored.evaluation.failed_queries == run.evaluation.failed_queries
        assert restored.evaluation.mean_latency_ms == run.evaluation.mean_latency_ms
        assert restored.evaluation.abstention_accuracy == run.evaluation.abstention_accuracy
        assert (
            restored.evaluation.abstention_evaluated_queries
            == run.evaluation.abstention_evaluated_queries
        )

        assert restored.quality_gate.passed == run.quality_gate.passed
        assert restored.quality_gate.errors == run.quality_gate.errors
        assert restored.quality_gate.policy == run.quality_gate.policy
        assert restored.external_evaluations == run.external_evaluations

        assert restored.external_quality_gate is not None
        assert restored.external_quality_gate.passed is True
        assert restored.external_quality_gate.errors == ()
        assert restored.external_quality_gate.metrics == {
            "ragas/faithfulness/faithfulness": 0.91,
        }
        assert restored.external_quality_gate.policy == (run.external_quality_gate.policy)
    finally:
        if repository is not None:
            repository._session.execute(
                text("DELETE FROM retrieval_evaluation_runs " "WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            repository._session.commit()
            repository._session.close()

        if engine is not None:
            engine.dispose()
