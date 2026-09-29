from datetime import UTC, datetime

import pytest

from ai_platform.agents.evaluation.answer_evaluation import AgentAnswerEvaluator
from ai_platform.agents.evaluation.evaluator import AgentEvaluator
from ai_platform.agents.evaluation.models import AgentRunEvidence
from ai_platform.agents.evaluation.policy import AgentEvaluationPolicy
from ai_platform.agents.evaluation.run import (
    AgentEvaluationLineage,
    AgentEvaluationRun,
)


def _evidence(**overrides) -> AgentRunEvidence:
    values = {
        "run_id": "run-1",
        "agent_name": "test-agent",
        "agent_version": "v1",
        "tenant_id": "tenant-1",
        "status": "completed",
        "started_at": datetime.now(UTC),
        "completed_at": datetime.now(UTC),
        "execution_time_ms": 100.0,
        "total_steps": 3,
        "tool_calls_total": 2,
        "tool_calls_successful": 2,
        "tool_calls_failed": 0,
        "invalid_tool_calls": 0,
        "governance_denials": 0,
        "rag_queries_total": 2,
        "rag_sources_retrieved_total": 5,
        "has_final_answer": True,
        "final_answer_length": 42,
        "rag_sources_available_count": 5,
        "rag_unique_chunks_count": 4,
    }
    values.update(overrides)
    return AgentRunEvidence(**values)


def test_evaluator_produces_deterministic_metrics_and_passes_gate():
    policy = AgentEvaluationPolicy(
        max_execution_time_ms=500,
        max_steps_per_run=5,
        max_invalid_tool_calls=0,
    )

    metrics, gate = AgentEvaluator.evaluate_run(_evidence(), policy)

    assert metrics.execution_time_ms == 100.0
    assert metrics.steps_total == 3
    assert metrics.tool_calls_total == 2
    assert metrics.tool_calls_successful == 2
    assert metrics.rag_queries_total == 2
    assert metrics.rag_sources_retrieved_total == 5
    assert metrics.has_final_answer is True
    assert metrics.final_answer_length == 42
    assert metrics.rag_sources_available_count == 5
    assert metrics.rag_unique_chunks_count == 4
    assert metrics.task_completed is True
    assert gate.passed is True
    assert gate.violations == ()


def test_answer_mismatch_does_not_fail_gate_when_not_required():
    policy = AgentEvaluationPolicy(require_answer_match=False)
    answer_evaluation = AgentAnswerEvaluator.evaluate(
        actual_answer="Actual response",
        expected_answer="Expected response",
    )

    _, gate = AgentEvaluator.evaluate_run(
        _evidence(),
        policy,
        answer_evaluation=answer_evaluation,
    )

    assert answer_evaluation.evaluated is True
    assert answer_evaluation.exact_match is False
    assert gate.passed is True
    assert gate.violations == ()


def test_required_answer_match_passes_when_answer_matches():
    policy = AgentEvaluationPolicy(require_answer_match=True)
    answer_evaluation = AgentAnswerEvaluator.evaluate(
        actual_answer="Expected response",
        expected_answer="expected   response",
    )

    _, gate = AgentEvaluator.evaluate_run(
        _evidence(),
        policy,
        answer_evaluation=answer_evaluation,
    )

    assert answer_evaluation.evaluated is True
    assert answer_evaluation.exact_match is True
    assert gate.passed is True
    assert gate.violations == ()


def test_required_answer_match_rejects_mismatched_answer():
    policy = AgentEvaluationPolicy(require_answer_match=True)
    answer_evaluation = AgentAnswerEvaluator.evaluate(
        actual_answer="Actual response",
        expected_answer="Expected response",
    )

    _, gate = AgentEvaluator.evaluate_run(
        _evidence(),
        policy,
        answer_evaluation=answer_evaluation,
    )

    assert answer_evaluation.evaluated is True
    assert answer_evaluation.exact_match is False
    assert gate.passed is False
    assert gate.violations == ("Final answer did not match the expected answer.",)


def test_required_answer_match_rejects_missing_evaluation():
    policy = AgentEvaluationPolicy(require_answer_match=True)

    _, gate = AgentEvaluator.evaluate_run(
        _evidence(),
        policy,
    )

    assert gate.passed is False
    assert gate.violations == (
        "Answer match was required but answer evaluation was not performed.",
    )


def test_quality_gate_rejects_multiple_violations():
    policy = AgentEvaluationPolicy(
        max_execution_time_ms=50,
        max_steps_per_run=2,
        max_invalid_tool_calls=0,
        allow_governance_denials=False,
    )

    metrics, gate = AgentEvaluator.evaluate_run(
        _evidence(
            execution_time_ms=100,
            total_steps=3,
            invalid_tool_calls=1,
            governance_denials=1,
        ),
        policy,
    )

    assert metrics.task_completed is True
    assert gate.passed is False
    assert len(gate.violations) == 4


def test_evaluator_produces_context_quality_assessment():
    policy = AgentEvaluationPolicy()

    metrics, gate = AgentEvaluator.evaluate_run(
        _evidence(
            context_assembly_events_total=2,
            context_messages_total=10,
            context_estimated_tokens_total=36,
            context_estimated_tokens_max=24,
            context_budget_exceeded=False,
            context_source_counts={
                "system_prompt": 2,
                "semantic_memory": 2,
                "user_input": 1,
                "chat_history": 3,
                "tool_result": 2,
            },
        ),
        policy,
    )

    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=2,
            context_messages_total=10,
            context_estimated_tokens_total=36,
            context_estimated_tokens_max=24,
            context_budget_exceeded=False,
            context_source_counts={
                "system_prompt": 2,
                "semantic_memory": 2,
                "user_input": 1,
                "chat_history": 3,
                "tool_result": 2,
            },
        )
    )

    assert metrics.task_completed is True
    assert gate.passed is True
    assert context_quality.budget_compliant is True
    assert context_quality.assemblies_total == 2
    assert context_quality.messages_total == 10
    assert context_quality.estimated_tokens_total == 36
    assert context_quality.estimated_tokens_max == 24
    assert context_quality.source_counts == {
        "system_prompt": 2,
        "semantic_memory": 2,
        "user_input": 1,
        "chat_history": 3,
        "tool_result": 2,
    }


def test_evaluator_context_quality_reports_budget_violation():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=2,
            context_messages_total=10,
            context_estimated_tokens_total=36,
            context_estimated_tokens_max=24,
            context_budget_exceeded=True,
        )
    )

    assert context_quality.budget_compliant is False


def test_evaluator_context_quality_is_unavailable_without_context_events():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=0,
            context_messages_total=0,
            context_estimated_tokens_total=0,
            context_estimated_tokens_max=0,
            context_budget_exceeded=False,
        )
    )

    assert context_quality.budget_compliant is None
    assert context_quality.assemblies_total == 0
    assert context_quality.messages_total == 0
    assert context_quality.estimated_tokens_total == 0
    assert context_quality.estimated_tokens_max == 0
    assert context_quality.source_counts == {}


def test_evaluator_context_quality_reports_retrieval_evidence_presence():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=2,
            context_source_counts={
                "system_prompt": 2,
                "semantic_memory": 3,
                "chat_history": 2,
                "user_input": 1,
            },
        )
    )

    assert context_quality.retrieval_evidence_present is True


def test_evaluator_context_quality_accepts_episodic_retrieval_evidence():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "system_prompt": 1,
                "episodic_memory": 2,
                "user_input": 1,
            },
        )
    )

    assert context_quality.retrieval_evidence_present is True


def test_evaluator_context_quality_reports_semantic_memory_presence():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "system_prompt": 1,
                "semantic_memory": 3,
                "user_input": 1,
            },
        )
    )

    assert context_quality.has_semantic_memory_sources is True
    assert context_quality.has_episodic_memory_sources is False


def test_evaluator_context_quality_reports_episodic_memory_presence():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "system_prompt": 1,
                "episodic_memory": 2,
                "user_input": 1,
            },
        )
    )

    assert context_quality.has_semantic_memory_sources is False
    assert context_quality.has_episodic_memory_sources is True


def test_evaluator_context_quality_reports_both_memory_sources():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "semantic_memory": 2,
                "episodic_memory": 1,
            },
        )
    )

    assert context_quality.has_semantic_memory_sources is True
    assert context_quality.has_episodic_memory_sources is True


def test_evaluator_context_quality_reports_no_retrieval_memory_sources():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "system_prompt": 1,
                "chat_history": 2,
                "user_input": 1,
            },
        )
    )

    assert context_quality.has_semantic_memory_sources is False
    assert context_quality.has_episodic_memory_sources is False


def test_evaluator_context_quality_reports_source_profile_changes():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=4,
            context_source_profile_changes=2,
            context_source_counts={
                "system_prompt": 4,
                "semantic_memory": 4,
                "user_input": 4,
                "tool_result": 3,
            },
        )
    )

    assert context_quality.context_source_profile_changes == 2


def test_evaluator_context_quality_reports_zero_source_profile_changes():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_profile_changes=0,
        )
    )

    assert context_quality.context_source_profile_changes == 0


def test_evaluator_context_quality_reports_working_memory_presence():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "system_prompt": 1,
                "working_memory": 2,
                "user_input": 1,
            },
        )
    )

    assert context_quality.has_working_memory_sources is True
    assert context_quality.has_chat_history_sources is False
    assert context_quality.has_tool_result_sources is False


def test_evaluator_context_quality_reports_chat_history_presence():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "system_prompt": 1,
                "chat_history": 3,
                "user_input": 1,
            },
        )
    )

    assert context_quality.has_working_memory_sources is False
    assert context_quality.has_chat_history_sources is True
    assert context_quality.has_tool_result_sources is False


def test_evaluator_context_quality_reports_tool_result_presence():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "system_prompt": 1,
                "tool_result": 2,
                "user_input": 1,
            },
        )
    )

    assert context_quality.has_working_memory_sources is False
    assert context_quality.has_chat_history_sources is False
    assert context_quality.has_tool_result_sources is True


def test_evaluator_context_quality_reports_multiple_context_components():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "working_memory": 2,
                "chat_history": 3,
                "tool_result": 1,
            },
        )
    )

    assert context_quality.has_working_memory_sources is True
    assert context_quality.has_chat_history_sources is True
    assert context_quality.has_tool_result_sources is True


def test_evaluator_context_quality_reports_missing_retrieval_evidence():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts={
                "system_prompt": 1,
                "chat_history": 2,
                "user_input": 1,
            },
        )
    )

    assert context_quality.retrieval_evidence_present is False


def test_evaluator_context_quality_retrieval_evidence_is_unavailable_without_context():
    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=0,
            context_source_counts={},
        )
    )

    assert context_quality.retrieval_evidence_present is None
    assert context_quality.has_semantic_memory_sources is None
    assert context_quality.has_episodic_memory_sources is None
    assert context_quality.has_working_memory_sources is None
    assert context_quality.has_chat_history_sources is None
    assert context_quality.has_tool_result_sources is None


def test_evaluator_context_quality_copies_source_counts():
    source_counts = {
        "semantic_memory": 2,
        "tool_result": 1,
    }

    context_quality = AgentEvaluator.evaluate_context(
        _evidence(
            context_assembly_events_total=1,
            context_source_counts=source_counts,
        )
    )

    source_counts["semantic_memory"] = 99
    source_counts["new_source"] = 7

    assert context_quality.source_counts == {
        "semantic_memory": 2,
        "tool_result": 1,
    }


def test_quality_gate_passes_rag_score_thresholds():
    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
    )

    metrics, gate = AgentEvaluator.evaluate_run(
        _evidence(
            retrieval_score_avg=0.72,
            reranker_score_avg=0.91,
        ),
        policy,
    )

    assert metrics.retrieval_score_avg == 0.72
    assert metrics.reranker_score_avg == 0.91
    assert gate.passed is True
    assert gate.violations == ()


def test_quality_gate_rejects_rag_score_thresholds():
    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
    )

    _, gate = AgentEvaluator.evaluate_run(
        _evidence(
            retrieval_score_avg=0.61,
            reranker_score_avg=0.84,
        ),
        policy,
    )

    assert gate.passed is False
    assert gate.violations == (
        "Average retrieval score (0.61) was below " "minimum threshold (0.70).",
        "Average reranker score (0.84) was below " "minimum threshold (0.90).",
    )


def test_quality_gate_rejects_missing_rag_scores_when_threshold_required():
    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
    )

    _, gate = AgentEvaluator.evaluate_run(
        _evidence(),
        policy,
    )

    assert gate.passed is False
    assert gate.violations == (
        "Average retrieval score was unavailable but minimum threshold " "(0.70) is required.",
        "Average reranker score was unavailable but minimum threshold " "(0.90) is required.",
    )


def test_quality_gate_ignores_rag_scores_without_thresholds():
    policy = AgentEvaluationPolicy()

    _, gate = AgentEvaluator.evaluate_run(
        _evidence(
            retrieval_score_avg=0.20,
            reranker_score_avg=0.10,
        ),
        policy,
    )

    assert gate.passed is True
    assert gate.violations == ()


def test_evaluation_policy_serializes_rag_score_thresholds():
    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
    )

    assert policy.as_dict()["min_retrieval_score"] == 0.70
    assert policy.as_dict()["min_reranker_score"] == 0.90


def test_evaluation_policy_rejects_negative_rag_score_thresholds():
    with pytest.raises(ValueError, match="min_retrieval_score must be non-negative"):
        AgentEvaluationPolicy(min_retrieval_score=-0.01)

    with pytest.raises(ValueError, match="min_reranker_score must be non-negative"):
        AgentEvaluationPolicy(min_reranker_score=-0.01)


def test_quality_gate_rejects_incomplete_run():
    policy = AgentEvaluationPolicy()

    metrics, gate = AgentEvaluator.evaluate_run(
        _evidence(status="failed"),
        policy,
    )

    assert metrics.task_completed is False
    assert gate.passed is False
    assert gate.violations == ("Agent task execution did not complete successfully.",)


def test_governance_denials_can_be_explicitly_allowed():
    policy = AgentEvaluationPolicy(
        allow_governance_denials=True,
    )

    _, gate = AgentEvaluator.evaluate_run(
        _evidence(governance_denials=2),
        policy,
    )

    assert gate.passed is True


def test_evaluation_run_is_immutable_and_serializable():
    policy = AgentEvaluationPolicy(max_steps_per_run=5)
    metrics, gate = AgentEvaluator.evaluate_run(_evidence(), policy)

    evaluation = AgentEvaluationRun(
        evaluation_run_id="evaluation-1",
        created_at=datetime.now(UTC),
        lineage=AgentEvaluationLineage(
            evaluated_run_id="run-1",
            agent_name="test-agent",
            agent_version="v1",
            tenant_id="tenant-1",
        ),
        metrics=metrics,
        policy=policy,
        quality_gate=gate,
    )

    assert evaluation.passed is True
    payload = evaluation.as_dict()
    assert payload["evaluation_run_id"] == "evaluation-1"
    assert payload["lineage"]["evaluated_run_id"] == "run-1"
    assert payload["metrics"]["steps_total"] == 3

    with pytest.raises(AttributeError):
        evaluation.evaluation_run_id = "changed"


@pytest.mark.parametrize(
    "field",
    [
        "execution_time_ms",
        "total_steps",
        "tool_calls_total",
        "tool_calls_successful",
        "tool_calls_failed",
        "invalid_tool_calls",
        "governance_denials",
        "rag_queries_total",
        "rag_sources_retrieved_total",
    ],
)
def test_evidence_rejects_negative_metrics(field):
    with pytest.raises(ValueError):
        _evidence(**{field: -1})


def test_evaluation_policy_serializes_identity_and_version():
    policy = AgentEvaluationPolicy(
        policy_id="rag-quality",
        policy_version="1.0",
        name="RAG Quality Policy",
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
    )

    assert policy.as_dict()["policy_id"] == "rag-quality"
    assert policy.as_dict()["policy_version"] == "1.0"
    assert policy.as_dict()["name"] == "RAG Quality Policy"


def test_evaluation_policy_rejects_blank_identity():
    with pytest.raises(ValueError, match="policy_id"):
        AgentEvaluationPolicy(policy_id="   ")

    with pytest.raises(ValueError, match="policy_id"):
        AgentEvaluationPolicy(policy_id="")


def test_evaluation_policy_rejects_blank_version():
    with pytest.raises(ValueError, match="policy_version"):
        AgentEvaluationPolicy(policy_version="   ")

    with pytest.raises(ValueError, match="policy_version"):
        AgentEvaluationPolicy(policy_version="")


def test_evaluation_policy_allows_legacy_missing_identity():
    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        name="legacy-policy",
    )

    assert policy.policy_id is None
    assert policy.policy_version is None

    serialized = policy.as_dict()
    assert serialized["policy_id"] is None
    assert serialized["policy_version"] is None


def test_evaluation_policy_rejects_identity_without_version():
    with pytest.raises(
        ValueError,
        match="policy_id and policy_version must be provided together",
    ):
        AgentEvaluationPolicy(policy_id="rag-quality")


def test_evaluation_policy_rejects_version_without_identity():
    with pytest.raises(
        ValueError,
        match="policy_id and policy_version must be provided together",
    ):
        AgentEvaluationPolicy(policy_version="1.0")


def test_evaluation_policy_allows_versioned_identity_pair():
    policy = AgentEvaluationPolicy(
        policy_id="rag-quality",
        policy_version="1.0",
    )

    assert policy.policy_id == "rag-quality"
    assert policy.policy_version == "1.0"
