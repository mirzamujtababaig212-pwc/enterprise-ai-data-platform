from datetime import UTC, datetime, timedelta

import pytest

from ai_platform.agents.evaluation.policy import AgentEvaluationPolicy
from app.control_plane.agent_evaluations.application_service import (
    AgentEvaluationApplicationService,
)
from app.control_plane.agent_evaluations.in_memory import (
    InMemoryAgentEvaluationRunsRepository,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)


class FakeAgentRunRepository:
    def __init__(self, run: AgentRun | None) -> None:
        self.run = run
        self.requested_tenant_id: str | None = None

    def get_for_tenant(
        self,
        run_id: str,
        tenant_id: str,
    ) -> AgentRun | None:
        self.requested_tenant_id = tenant_id

        if self.run is None or self.run.run_id != run_id:
            return None

        if self.run.tenant_id != tenant_id:
            return None

        return self.run


class FakeAgentRunStepsRepository:
    def __init__(self, steps: list[AgentRunStep]) -> None:
        self.steps = steps
        self.requested_limit: int | None = None

    def list(
        self,
        run_id: str,
        *,
        status=None,
        limit: int = 100,
    ) -> list[AgentRunStep]:
        self.requested_limit = limit
        return [step for step in self.steps if step.run_id == run_id]


class FakeAgentRunEventsRepository:
    def __init__(self, events: list[AgentExecutionEvent]) -> None:
        self.events = events
        self.requested_limit: int | None = None

    def list(
        self,
        run_id: str,
        *,
        limit: int = 100,
    ) -> list[AgentExecutionEvent]:
        self.requested_limit = limit
        return [event for event in self.events if event.run_id == run_id]


def make_run(
    *,
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
    tenant_id: str = "tenant-1",
    principal: str = "user-1",
) -> AgentRun:
    started_at = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    completed_at = started_at + timedelta(seconds=2)

    return AgentRun(
        run_id="run-1",
        agent_name="vehicle-agent",
        tenant_id=tenant_id,
        principal=principal,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        metadata={"agent_version": "1.2.3"},
    )


def make_step(
    step_id: str,
    *,
    status: AgentRunStepStatus = AgentRunStepStatus.COMPLETED,
    tool_name: str | None = "lookup_vehicle",
    call_id: str | None = "call-1",
    failure_category: str | None = None,
) -> AgentRunStep:
    return AgentRunStep(
        run_id="run-1",
        step_id=step_id,
        step_index=0,
        step_type="tool",
        status=status,
        tool_name=tool_name,
        call_id=call_id,
        failure_category=failure_category,
    )


def make_governance_denial() -> AgentExecutionEvent:
    return AgentExecutionEvent(
        event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
        agent_name="vehicle-agent",
        run_id="run-1",
        metadata={
            "governance_domain": "tool",
            "decision": "deny",
            "reason": "policy denied tool execution",
        },
    )


def make_service(
    *,
    run: AgentRun | None = None,
    steps: list[AgentRunStep] | None = None,
    events: list[AgentExecutionEvent] | None = None,
):
    run_repository = FakeAgentRunRepository(run or make_run())
    steps_repository = FakeAgentRunStepsRepository(steps or [])
    events_repository = FakeAgentRunEventsRepository(events or [])
    evaluation_repository = InMemoryAgentEvaluationRunsRepository()

    service = AgentEvaluationApplicationService(
        agent_run_repository=run_repository,
        agent_run_steps_repository=steps_repository,
        agent_run_events_repository=events_repository,
        evaluation_repository=evaluation_repository,
    )

    return (
        service,
        run_repository,
        steps_repository,
        events_repository,
        evaluation_repository,
    )


def default_policy() -> AgentEvaluationPolicy:
    return AgentEvaluationPolicy(
        max_execution_time_ms=5_000,
        max_steps_per_run=10,
        max_invalid_tool_calls=0,
        allow_governance_denials=False,
        require_task_completed=True,
        name="default-agent-quality",
    )


def test_evaluate_run_builds_and_persists_immutable_artifact():
    service, _, _, _, repository = make_service(
        steps=[
            make_step("step-1"),
            make_step("step-2", tool_name=None, call_id=None),
        ]
    )

    result = service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.lineage.evaluated_run_id == "run-1"
    assert result.lineage.agent_name == "vehicle-agent"
    assert result.lineage.agent_version == "1.2.3"
    assert result.lineage.tenant_id == "tenant-1"
    assert result.metrics.steps_total == 2
    assert result.metrics.tool_calls_total == 1
    assert result.metrics.tool_calls_successful == 1
    assert result.metrics.tool_calls_failed == 0
    assert result.metrics.invalid_tool_calls == 0
    assert result.metrics.governance_denials == 0
    assert result.metrics.task_completed is True
    assert result.passed is True
    assert repository.get(result.evaluation_run_id) == result


def test_evaluate_run_counts_governance_denials():
    service, _, _, _, _ = make_service(events=[make_governance_denial()])

    result = service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.metrics.governance_denials == 1
    assert result.passed is False
    assert "governance denial" in result.quality_gate.violations[0]


def test_evaluate_run_preserves_failed_tool_metrics():
    service, _, _, _, _ = make_service(
        steps=[
            make_step(
                "step-1",
                status=AgentRunStepStatus.FAILED,
                failure_category="execution_error",
            ),
        ]
    )

    result = service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.metrics.tool_calls_total == 1
    assert result.metrics.tool_calls_successful == 0
    assert result.metrics.tool_calls_failed == 1


def test_evaluate_run_applies_rag_score_thresholds_to_durable_evidence():
    service, _, _, _, repository = make_service(
        steps=[
            AgentRunStep(
                run_id="run-1",
                step_id="step-rag",
                step_index=0,
                step_type="tool",
                status=AgentRunStepStatus.COMPLETED,
                tool_name="rag.search",
                call_id="call-rag-1",
                metadata={
                    "rag_provenance": {
                        "retrieved_count": 3,
                        "sources": [
                            {
                                "chunk_id": "chunk-1",
                                "retrieval_score": 0.72,
                                "reranker_score": 0.91,
                            },
                            {
                                "chunk_id": "chunk-2",
                                "retrieval_score": 0.61,
                                "reranker_score": 0.88,
                            },
                            {
                                "chunk_id": "chunk-3",
                                "retrieval_score": 0.83,
                                "reranker_score": 0.95,
                            },
                        ],
                    }
                },
            ),
        ]
    )

    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
        name="rag-quality-v1",
    )

    result = service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=policy,
    )

    assert result.metrics.retrieval_score_avg == pytest.approx((0.72 + 0.61 + 0.83) / 3)
    assert result.metrics.reranker_score_avg == pytest.approx((0.91 + 0.88 + 0.95) / 3)
    assert result.passed is True
    assert result.quality_gate.violations == ()
    assert repository.get(result.evaluation_run_id) == result


def test_evaluate_run_rejects_rag_score_thresholds_from_durable_evidence():
    service, _, _, _, repository = make_service(
        steps=[
            AgentRunStep(
                run_id="run-1",
                step_id="step-rag",
                step_index=0,
                step_type="tool",
                status=AgentRunStepStatus.COMPLETED,
                tool_name="rag.search",
                call_id="call-rag-1",
                metadata={
                    "rag_provenance": {
                        "retrieved_count": 3,
                        "sources": [
                            {
                                "chunk_id": "chunk-1",
                                "retrieval_score": 0.52,
                                "reranker_score": 0.91,
                            },
                            {
                                "chunk_id": "chunk-2",
                                "retrieval_score": 0.61,
                                "reranker_score": 0.88,
                            },
                            {
                                "chunk_id": "chunk-3",
                                "retrieval_score": 0.63,
                                "reranker_score": 0.95,
                            },
                        ],
                    }
                },
            ),
        ]
    )

    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
        name="rag-quality-v1",
    )

    result = service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=policy,
    )

    assert result.metrics.retrieval_score_avg == pytest.approx((0.52 + 0.61 + 0.63) / 3)
    assert result.metrics.reranker_score_avg == pytest.approx((0.91 + 0.88 + 0.95) / 3)
    assert result.passed is False
    assert any(
        "Average retrieval score" in violation for violation in result.quality_gate.violations
    )
    assert repository.get(result.evaluation_run_id) == result


def test_evaluate_run_rejects_missing_run():
    service, _, _, _, _ = make_service(run=None)

    with pytest.raises(
        LookupError,
        match="agent run not found: missing",
    ):
        service.evaluate_run(
            "missing",
            tenant_id="tenant-1",
            principal="user-1",
            policy=default_policy(),
        )


def test_evaluate_run_enforces_principal_authorization():
    service, _, _, _, _ = make_service(
        run=make_run(principal="different-user"),
    )

    with pytest.raises(
        PermissionError,
        match="principal is not authorized",
    ):
        service.evaluate_run(
            "run-1",
            tenant_id="tenant-1",
            principal="user-1",
            policy=default_policy(),
        )


def test_evaluate_run_uses_bounded_repository_reads():
    service, _, steps_repository, events_repository, _ = make_service()

    service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert steps_repository.requested_limit == 10_000
    assert events_repository.requested_limit == 10_000
