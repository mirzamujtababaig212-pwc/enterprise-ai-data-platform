from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
            rag_sources_available_count=5,
            rag_unique_chunks_count=4,
            retrieval_score_min=0.61,
            retrieval_score_max=0.83,
            retrieval_score_avg=0.72,
            reranker_score_min=0.88,
            reranker_score_max=0.95,
            reranker_score_avg=0.91,
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
            name="default-agent-quality",
        ),
        quality_gate=AgentQualityGateResult(
            passed=True,
            violations=(),
        ),
    )


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
        assert restored.metrics.retrieval_score_min == run.metrics.retrieval_score_min
        assert restored.metrics.retrieval_score_max == run.metrics.retrieval_score_max
        assert restored.metrics.retrieval_score_avg == run.metrics.retrieval_score_avg
        assert restored.metrics.reranker_score_min == run.metrics.reranker_score_min
        assert restored.metrics.reranker_score_max == run.metrics.reranker_score_max
        assert restored.metrics.reranker_score_avg == run.metrics.reranker_score_avg
        assert restored.policy.policy_id == run.policy.policy_id
        assert restored.policy.policy_version == run.policy.policy_version
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
        ):
            legacy_metrics.pop(key, None)

        record.metrics = legacy_metrics
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
