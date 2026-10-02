from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ai_platform.agents.evaluation.answer_evaluation import AgentAnswerEvaluation
from ai_platform.agents.evaluation.models import (
    AgentContextQualityAssessment,
    AgentEvaluationMetrics,
)
from ai_platform.agents.evaluation.policy import (
    AgentEvaluationPolicy,
    AgentQualityGateResult,
)
from ai_platform.agents.evaluation.run import (
    AgentEvaluationLineage,
    AgentEvaluationRun,
)
from app.control_plane.agent_evaluations.postgres_repository import (
    PostgreSQLAgentEvaluationRunsRepository,
)
from app.control_plane.persistence.models import AgentEvaluationRunRecord, Base
from rag.evaluation.lineage import RetrievalEvaluationArtifact


def _repository():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return PostgreSQLAgentEvaluationRunsRepository(session_factory()), engine


def _run(
    evaluation_run_id: str = "evaluation-score-diagnostics",
    *,
    include_context_quality: bool = True,
) -> AgentEvaluationRun:
    return AgentEvaluationRun(
        evaluation_run_id=evaluation_run_id,
        created_at=datetime(2026, 9, 28, 12, 0, tzinfo=UTC),
        lineage=AgentEvaluationLineage(
            evaluated_run_id="run-score-diagnostics",
            agent_name="vehicle-agent",
            agent_version="1.2.3",
            tenant_id="tenant-acme",
            evidence_fingerprint="b" * 64,
        ),
        metrics=AgentEvaluationMetrics(
            execution_time_ms=125.5,
            steps_total=3,
            tool_calls_total=2,
            tool_calls_successful=2,
            tool_calls_failed=0,
            invalid_tool_calls=0,
            governance_denials=0,
            task_completed=True,
            rag_queries_total=2,
            rag_sources_retrieved_total=5,
            has_final_answer=True,
            final_answer_length=42,
            has_rag_provenance=True,
            has_rag_sources_available=True,
            rag_sources_available_count=5,
            rag_unique_chunks_count=4,
            retrieval_score_min=0.61,
            retrieval_score_max=0.83,
            retrieval_score_avg=0.72,
            reranker_score_min=0.88,
            reranker_score_max=0.95,
            reranker_score_avg=0.91,
            grounding_evaluated=True,
            grounding_supported=True,
            grounding_support_ratio=1.0,
            grounding_supported_sources_total=2,
            grounding_source_candidates_total=5,
            grounding_method="lexical_sentence_support_v1",
            semantic_grounding_evaluated=True,
            semantic_grounding_score=0.93,
            semantic_grounding_passed=True,
            semantic_grounding_method="llm_grounding_judge_v1",
            semantic_grounding_evaluator_model="gpt-4.1-mini",
            semantic_grounding_evaluator_provider="openai",
        ),
        context_quality=(
            AgentContextQualityAssessment(
                budget_compliant=True,
                retrieval_evidence_present=True,
                has_semantic_memory_sources=True,
                has_episodic_memory_sources=False,
                has_working_memory_sources=True,
                has_chat_history_sources=True,
                has_tool_result_sources=True,
                context_source_profile_changes=2,
                assemblies_total=3,
                messages_total=12,
                estimated_tokens_total=900,
                estimated_tokens_max=400,
                minimum_estimated_remaining_after_context=120,
                source_counts={
                    "system_prompt": 3,
                    "semantic_memory": 3,
                    "working_memory": 3,
                    "chat_history": 3,
                    "user_input": 3,
                    "tool_result": 2,
                },
            )
            if include_context_quality
            else None
        ),
        policy=AgentEvaluationPolicy(
            policy_id="rag-quality",
            policy_version="1.0",
            max_execution_time_ms=1000,
            max_steps_per_run=10,
            max_invalid_tool_calls=0,
            allow_governance_denials=False,
            require_task_completed=True,
            require_rag_provenance=True,
            min_grounding_support_ratio=0.88,
            min_semantic_grounding_score=0.80,
            name="default-agent-quality",
        ),
        quality_gate=AgentQualityGateResult(
            passed=True,
            violations=(),
        ),
    )


def test_save_and_get_round_trip_preserves_evidence_fingerprint():
    repository, engine = _repository()

    try:
        run = _run("evaluation-evidence-fingerprint")

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )
        assert record is not None

        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.lineage.evidence_fingerprint == "b" * 64
        assert restored.lineage.evidence_fingerprint == run.lineage.evidence_fingerprint

        assert record.lineage["evidence_fingerprint"] == "b" * 64
    finally:
        repository.close()
        engine.dispose()


def test_save_and_get_round_trip_preserves_rag_score_diagnostics():
    repository, engine = _repository()

    try:
        run = _run()

        repository.save(run)

        # SQLite does not preserve timezone information for DateTime(timezone=True).
        # Restore the domain-valid timestamp so this test isolates JSON metric
        # persistence rather than SQLAlchemy/SQLite timestamp behavior.
        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )
        assert record is not None
        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.metrics.has_rag_provenance is True
        assert restored.metrics.has_rag_sources_available is True
        assert (
            restored.metrics.rag_sources_available_count == run.metrics.rag_sources_available_count
        )
        assert restored.policy.require_rag_provenance is True
        assert restored.policy.min_grounding_support_ratio == 0.88
        assert restored.policy.min_semantic_grounding_score == 0.80
        assert restored.metrics.retrieval_score_min == run.metrics.retrieval_score_min
        assert restored.metrics.retrieval_score_max == run.metrics.retrieval_score_max
        assert restored.metrics.retrieval_score_avg == run.metrics.retrieval_score_avg
        assert restored.metrics.reranker_score_min == run.metrics.reranker_score_min
        assert restored.metrics.reranker_score_max == run.metrics.reranker_score_max
        assert restored.metrics.reranker_score_avg == run.metrics.reranker_score_avg
        assert restored.metrics.grounding_evaluated is True
        assert restored.metrics.grounding_supported is True
        assert restored.metrics.grounding_support_ratio == 1.0
        assert (
            restored.metrics.grounding_supported_sources_total
            == run.metrics.grounding_supported_sources_total
        )
        assert (
            restored.metrics.grounding_source_candidates_total
            == run.metrics.grounding_source_candidates_total
        )
        assert restored.metrics.grounding_method == run.metrics.grounding_method
        assert restored.metrics.semantic_grounding_evaluated is True
        assert restored.metrics.semantic_grounding_score == 0.93
        assert restored.metrics.semantic_grounding_passed is True
        assert restored.metrics.semantic_grounding_method == "llm_grounding_judge_v1"
        assert restored.metrics.semantic_grounding_evaluator_model == "gpt-4.1-mini"
        assert restored.metrics.semantic_grounding_evaluator_provider == "openai"
        assert restored.policy.policy_id == run.policy.policy_id
        assert restored.policy.policy_version == run.policy.policy_version
    finally:
        repository.close()
        engine.dispose()


def test_get_legacy_run_without_evidence_fingerprint_preserves_compatibility():
    repository, engine = _repository()

    try:
        run = _run("legacy-evidence-fingerprint")

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )

        assert record is not None

        legacy_lineage = dict(record.lineage)
        legacy_lineage.pop("evidence_fingerprint", None)
        record.lineage = legacy_lineage

        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.lineage.evidence_fingerprint is None
        assert restored.lineage.evaluated_run_id == run.lineage.evaluated_run_id
        assert restored.lineage.agent_name == run.lineage.agent_name
        assert restored.lineage.tenant_id == run.lineage.tenant_id
    finally:
        repository.close()
        engine.dispose()


def test_get_legacy_run_without_score_diagnostics_preserves_compatibility():
    repository, engine = _repository()

    try:
        run = _run("legacy-score-diagnostics")

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )

        assert record is not None

        legacy_metrics = dict(record.metrics)
        for key in (
            "retrieval_score_min",
            "retrieval_score_max",
            "retrieval_score_avg",
            "reranker_score_min",
            "reranker_score_max",
            "reranker_score_avg",
            "has_rag_provenance",
            "has_rag_sources_available",
        ):
            legacy_metrics.pop(key, None)

        record.metrics = legacy_metrics

        legacy_policy = dict(record.policy)
        legacy_policy.pop("require_rag_provenance", None)
        legacy_policy.pop("min_semantic_grounding_score", None)
        record.policy = legacy_policy

        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.metrics.retrieval_score_min is None
        assert restored.metrics.retrieval_score_max is None
        assert restored.metrics.retrieval_score_avg is None
        assert restored.metrics.reranker_score_min is None
        assert restored.metrics.reranker_score_max is None
        assert restored.metrics.reranker_score_avg is None
        assert restored.metrics.has_rag_provenance is False
        assert restored.metrics.has_rag_sources_available is False
        assert restored.policy.require_rag_provenance is False
        assert restored.policy.min_semantic_grounding_score is None
    finally:
        repository.close()
        engine.dispose()


def test_get_legacy_run_without_grounding_diagnostics_preserves_compatibility():
    repository, engine = _repository()

    try:
        run = _run("legacy-grounding-diagnostics")

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )

        assert record is not None

        legacy_metrics = dict(record.metrics)
        for key in (
            "grounding_evaluated",
            "grounding_supported",
            "grounding_support_ratio",
            "grounding_supported_sources_total",
            "grounding_source_candidates_total",
            "grounding_method",
        ):
            legacy_metrics.pop(key, None)

        record.metrics = legacy_metrics
        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.metrics.grounding_evaluated is False
        assert restored.metrics.grounding_supported is None
        assert restored.metrics.grounding_support_ratio is None
        assert restored.metrics.grounding_supported_sources_total == 0
        assert restored.metrics.grounding_source_candidates_total == 0
        assert restored.metrics.grounding_method is None
    finally:
        repository.close()
        engine.dispose()


def test_save_and_get_round_trip_preserves_semantic_answer_evaluation():
    repository, engine = _repository()

    try:
        run = AgentEvaluationRun(
            evaluation_run_id="evaluation-semantic-answer",
            created_at=datetime(2026, 9, 28, 12, 0, tzinfo=UTC),
            lineage=AgentEvaluationLineage(
                evaluated_run_id="run-semantic-answer",
                agent_name="vehicle-agent",
                agent_version="1.2.3",
                tenant_id="tenant-acme",
            ),
            metrics=_run().metrics,
            policy=_run().policy,
            quality_gate=_run().quality_gate,
            answer_evaluation=AgentAnswerEvaluation(
                evaluated=True,
                exact_match=True,
                normalization="whitespace_casefold",
                semantic_evaluated=True,
                semantic_score=0.92,
                semantic_passed=True,
                semantic_method="llm_judge_v1",
                evaluator_model="gpt-4.1-mini",
                evaluator_provider="openai",
            ),
        )

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )
        assert record is not None
        assert record.answer_evaluation == run.answer_evaluation.as_dict()

        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.answer_evaluation == run.answer_evaluation
        assert restored.answer_evaluation is not None
        assert restored.answer_evaluation.semantic_evaluated is True
        assert restored.answer_evaluation.semantic_score == 0.92
        assert restored.answer_evaluation.semantic_passed is True
        assert restored.answer_evaluation.semantic_method == "llm_judge_v1"
        assert restored.answer_evaluation.evaluator_model == "gpt-4.1-mini"
        assert restored.answer_evaluation.evaluator_provider == "openai"
    finally:
        repository.close()
        engine.dispose()


def test_get_legacy_answer_evaluation_preserves_compatibility():
    repository, engine = _repository()

    try:
        run = _run("legacy-answer-evaluation")

        run = AgentEvaluationRun(
            evaluation_run_id=run.evaluation_run_id,
            created_at=run.created_at,
            lineage=run.lineage,
            metrics=run.metrics,
            policy=run.policy,
            quality_gate=run.quality_gate,
            answer_evaluation=AgentAnswerEvaluation(
                evaluated=True,
                exact_match=True,
            ),
            context_quality=run.context_quality,
        )

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )

        assert record is not None

        legacy_answer_evaluation = dict(record.answer_evaluation)
        for key in (
            "semantic_evaluated",
            "semantic_score",
            "semantic_passed",
            "semantic_method",
            "evaluator_model",
            "evaluator_provider",
        ):
            legacy_answer_evaluation.pop(key, None)

        record.answer_evaluation = legacy_answer_evaluation
        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.answer_evaluation is not None
        assert restored.answer_evaluation.evaluated is True
        assert restored.answer_evaluation.exact_match is True
        assert restored.answer_evaluation.normalization == "whitespace_casefold"
        assert restored.answer_evaluation.semantic_evaluated is False
        assert restored.answer_evaluation.semantic_score is None
        assert restored.answer_evaluation.semantic_passed is None
        assert restored.answer_evaluation.semantic_method is None
        assert restored.answer_evaluation.evaluator_model is None
        assert restored.answer_evaluation.evaluator_provider is None
    finally:
        repository.close()
        engine.dispose()


def test_save_and_get_round_trip_preserves_context_quality():
    repository, engine = _repository()

    try:
        run = _run("evaluation-context-quality")

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )
        assert record is not None
        assert record.context_quality == run.context_quality.as_dict()

        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.context_quality == run.context_quality
        assert restored.context_quality is not None
        assert (
            restored.context_quality.minimum_estimated_remaining_after_context
            == run.context_quality.minimum_estimated_remaining_after_context
        )
        assert restored.context_quality.budget_compliant is True
        assert restored.context_quality.retrieval_evidence_present is True
        assert restored.context_quality.context_source_profile_changes == 2
        assert restored.context_quality.assemblies_total == 3
        assert restored.context_quality.estimated_tokens_total == 900
        assert restored.context_quality.source_counts == run.context_quality.source_counts
    finally:
        repository.close()
        engine.dispose()


def test_get_legacy_run_without_context_quality_preserves_compatibility():
    repository, engine = _repository()

    try:
        run = _run(
            "legacy-context-quality",
            include_context_quality=False,
        )

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )

        assert record is not None
        assert record.context_quality is None

        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.context_quality is None
    finally:
        repository.close()
        engine.dispose()


def test_get_legacy_run_without_semantic_grounding_diagnostics_preserves_compatibility():
    repository, engine = _repository()

    try:
        run = _run("legacy-semantic-grounding-diagnostics")

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )

        assert record is not None

        legacy_metrics = dict(record.metrics)
        for key in (
            "semantic_grounding_evaluated",
            "semantic_grounding_score",
            "semantic_grounding_passed",
            "semantic_grounding_method",
            "semantic_grounding_evaluator_model",
            "semantic_grounding_evaluator_provider",
        ):
            legacy_metrics.pop(key, None)

        record.metrics = legacy_metrics
        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.metrics.semantic_grounding_evaluated is False
        assert restored.metrics.semantic_grounding_score is None
        assert restored.metrics.semantic_grounding_passed is None
        assert restored.metrics.semantic_grounding_method is None
        assert restored.metrics.semantic_grounding_evaluator_model is None
        assert restored.metrics.semantic_grounding_evaluator_provider is None
    finally:
        repository.close()
        engine.dispose()


def test_save_and_get_round_trip_preserves_retrieval_artifact_and_legacy_compatibility():
    repository, engine = _repository()

    try:
        artifact = RetrievalEvaluationArtifact(
            retriever_type="HybridRetriever",
            vector_store_type="QdrantVectorStore",
        )

        run = _run("evaluation-retrieval-artifact")
        run = AgentEvaluationRun(
            evaluation_run_id=run.evaluation_run_id,
            created_at=run.created_at,
            lineage=AgentEvaluationLineage(
                evaluated_run_id=run.lineage.evaluated_run_id,
                agent_name=run.lineage.agent_name,
                agent_version=run.lineage.agent_version,
                tenant_id=run.lineage.tenant_id,
                effective_model=run.lineage.effective_model,
                effective_provider=run.lineage.effective_provider,
                model_policy_id=run.lineage.model_policy_id,
                model_policy_version=run.lineage.model_policy_version,
                evidence_fingerprint=run.lineage.evidence_fingerprint,
                retrieval_artifact=artifact,
            ),
            metrics=run.metrics,
            policy=run.policy,
            quality_gate=run.quality_gate,
            answer_evaluation=run.answer_evaluation,
            context_quality=run.context_quality,
        )

        repository.save(run)

        record = repository._session.get(
            AgentEvaluationRunRecord,
            run.evaluation_run_id,
        )
        assert record is not None
        assert record.lineage["retrieval_artifact"] == artifact.as_dict()

        record.created_at = run.created_at
        repository._session.flush()

        restored = repository.get(run.evaluation_run_id)

        assert restored is not None
        assert restored.lineage.retrieval_artifact == artifact

        # Simulate an older persisted evaluation whose lineage predates
        # retrieval-artifact persistence.
        legacy_lineage = dict(record.lineage)
        legacy_lineage.pop("retrieval_artifact", None)
        record.lineage = legacy_lineage
        repository._session.flush()

        legacy_restored = repository.get(run.evaluation_run_id)

        assert legacy_restored is not None
        assert legacy_restored.lineage.retrieval_artifact is None
    finally:
        repository.close()
        engine.dispose()
