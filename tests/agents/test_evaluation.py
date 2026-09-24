from datetime import UTC, datetime

import pytest

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
    assert metrics.task_completed is True
    assert gate.passed is True
    assert gate.violations == ()


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
    ],
)
def test_evidence_rejects_negative_metrics(field):
    with pytest.raises(ValueError):
        _evidence(**{field: -1})
