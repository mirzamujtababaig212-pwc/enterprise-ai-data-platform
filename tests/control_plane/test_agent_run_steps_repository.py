from datetime import UTC, datetime, timezone

import pytest

from app.control_plane.agent_run_steps.exceptions import (
    DuplicateAgentRunStepError,
    InvalidAgentRunStepTransitionError,
)
from app.control_plane.agent_run_steps.in_memory import (
    InMemoryAgentRunStepsRepository,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)


def make_step(
    *,
    run_id: str = "run-1",
    step_id: str = "step-1",
    step_index: int = 0,
    step_type: str = "tool",
) -> AgentRunStep:
    return AgentRunStep(
        run_id=run_id,
        step_id=step_id,
        step_index=step_index,
        step_type=step_type,
    )


def test_step_defaults_to_planned():
    step = make_step()

    assert step.status == AgentRunStepStatus.PLANNED
    assert step.attempt == 1
    assert step.metadata == {}


def test_valid_step_lifecycle():
    step = make_step()

    running = step.transition_to(AgentRunStepStatus.RUNNING)
    completed = running.transition_to(AgentRunStepStatus.COMPLETED)

    assert running.status == AgentRunStepStatus.RUNNING
    assert completed.status == AgentRunStepStatus.COMPLETED


def test_invalid_step_transition_is_rejected():
    step = make_step()

    with pytest.raises(InvalidAgentRunStepTransitionError):
        step.transition_to(AgentRunStepStatus.COMPLETED)


def test_failed_step_can_become_running_via_explicit_retry_transition():
    step = make_step().transition_to(AgentRunStepStatus.RUNNING)
    failed = step.transition_to(AgentRunStepStatus.FAILED)

    retried = failed.transition_to(AgentRunStepStatus.RUNNING)

    assert failed.status == AgentRunStepStatus.FAILED
    assert retried.status == AgentRunStepStatus.RUNNING
    assert retried.attempt == failed.attempt


def test_ambiguous_step_is_terminal():
    step = make_step().transition_to(AgentRunStepStatus.RUNNING)
    ambiguous = step.transition_to(AgentRunStepStatus.AMBIGUOUS)

    with pytest.raises(InvalidAgentRunStepTransitionError):
        ambiguous.transition_to(AgentRunStepStatus.RUNNING)


def test_in_memory_repository_create_get_and_list():
    repository = InMemoryAgentRunStepsRepository()

    step_2 = make_step(step_id="step-2", step_index=1)
    step_1 = make_step(step_id="step-1", step_index=0)

    repository.create(step_2)
    repository.create(step_1)

    assert repository.get("run-1", "step-1") == step_1
    assert repository.get("run-1", "missing") is None
    assert repository.list("run-1") == [step_1, step_2]


def test_in_memory_repository_rejects_duplicate_step():
    repository = InMemoryAgentRunStepsRepository()
    step = make_step()

    repository.create(step)

    with pytest.raises(DuplicateAgentRunStepError):
        repository.create(step)


def test_in_memory_repository_filters_by_status():
    repository = InMemoryAgentRunStepsRepository()

    step_1 = make_step(step_id="step-1", step_index=0)
    step_2 = make_step(step_id="step-2", step_index=1)

    repository.create(step_1)
    repository.create(step_2)

    now = datetime.now(timezone.utc)

    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=now,
        started_at=now,
    )

    running = repository.list(
        "run-1",
        status=AgentRunStepStatus.RUNNING,
    )

    assert running == [repository.get("run-1", "step-1")]


def test_in_memory_repository_transition_persists_execution_result():
    repository = InMemoryAgentRunStepsRepository()
    repository.create(make_step())

    now = datetime.now(timezone.utc)

    updated = repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=now,
        started_at=now,
    )

    assert updated is not None
    assert updated.status == AgentRunStepStatus.RUNNING
    assert updated.started_at == now

    completed = repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.COMPLETED,
        updated_at=now,
        completed_at=now,
        output={"result": "ok"},
    )

    assert completed is not None
    assert completed.status == AgentRunStepStatus.COMPLETED
    assert completed.output == {"result": "ok"}
    assert completed.completed_at == now


def test_in_memory_repository_retry_increments_attempt():
    repository = InMemoryAgentRunStepsRepository()
    repository.create(make_step())

    now = datetime.now(timezone.utc)

    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=now,
        started_at=now,
    )

    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.FAILED,
        updated_at=now,
        completed_at=now,
        error="temporary failure",
        failure_category="transient",
    )

    retried = repository.retry(
        "run-1",
        "step-1",
        updated_at=now,
        started_at=now,
    )

    assert retried is not None
    assert retried.status == AgentRunStepStatus.RUNNING
    assert retried.attempt == 2
    assert retried.output is None
    assert retried.error is None
    assert retried.failure_category is None
    assert retried.completed_at is None


def test_in_memory_repository_bind_execution_persists_tool_identity_and_input():
    from app.control_plane.agent_run_steps.in_memory import (
        InMemoryAgentRunStepsRepository,
    )

    repository = InMemoryAgentRunStepsRepository()

    repository.create(
        AgentRunStep(
            run_id="run-bind",
            step_id="retrieve_evidence",
            step_index=0,
            step_type="tool",
        )
    )

    started_at = datetime(2026, 9, 23, 10, 1, tzinfo=UTC)

    repository.transition(
        "run-bind",
        "retrieve_evidence",
        status=AgentRunStepStatus.RUNNING,
        updated_at=started_at,
        started_at=started_at,
    )

    updated = repository.bind_execution(
        "run-bind",
        "retrieve_evidence",
        tool_name="rag.search",
        call_id="call-123",
        input={"query": "RAG"},
    )

    assert updated is not None
    assert updated.status is AgentRunStepStatus.RUNNING
    assert updated.tool_name == "rag.search"
    assert updated.call_id == "call-123"
    assert updated.input == {"query": "RAG"}


def test_in_memory_repository_bind_execution_rejects_different_existing_binding():
    from app.control_plane.agent_run_steps.in_memory import (
        InMemoryAgentRunStepsRepository,
    )

    repository = InMemoryAgentRunStepsRepository()

    repository.create(
        AgentRunStep(
            run_id="run-bind",
            step_id="retrieve_evidence",
            step_index=0,
            step_type="tool",
        )
    )

    started_at = datetime(2026, 9, 23, 10, 1, tzinfo=UTC)

    repository.transition(
        "run-bind",
        "retrieve_evidence",
        status=AgentRunStepStatus.RUNNING,
        updated_at=started_at,
        started_at=started_at,
    )

    repository.bind_execution(
        "run-bind",
        "retrieve_evidence",
        tool_name="rag.search",
        call_id="call-123",
        input={"query": "RAG"},
    )

    with pytest.raises(ValueError, match="tool_name"):
        repository.bind_execution(
            "run-bind",
            "retrieve_evidence",
            tool_name="different.tool",
            call_id="call-123",
            input={"query": "RAG"},
        )


def test_in_memory_repository_bind_execution_rejects_different_input():
    repository = InMemoryAgentRunStepsRepository()

    step = AgentRunStep(
        run_id="run-1",
        step_id="step-1",
        step_index=0,
        step_type="tool",
    )
    repository.create(step)
    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=datetime.now(UTC),
    )

    repository.bind_execution(
        "run-1",
        "step-1",
        tool_name="vehicle.lookup",
        call_id="call-1",
        input={"vehicle_id": "V001"},
    )

    with pytest.raises(
        ValueError,
        match="agent run step input is already bound to a different value",
    ):
        repository.bind_execution(
            "run-1",
            "step-1",
            tool_name="vehicle.lookup",
            call_id="call-1",
            input={"vehicle_id": "V002"},
        )


def test_in_memory_repository_bind_execution_requires_running_status():
    from app.control_plane.agent_run_steps.in_memory import (
        InMemoryAgentRunStepsRepository,
    )

    repository = InMemoryAgentRunStepsRepository()

    repository.create(
        AgentRunStep(
            run_id="run-bind",
            step_id="retrieve_evidence",
            step_index=0,
            step_type="tool",
        )
    )

    with pytest.raises(ValueError, match="RUNNING"):
        repository.bind_execution(
            "run-bind",
            "retrieve_evidence",
            tool_name="rag.search",
            call_id="call-123",
            input={"query": "RAG"},
        )


def test_in_memory_repository_retry_requires_failed_status():
    repository = InMemoryAgentRunStepsRepository()
    repository.create(make_step())

    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="retry requires FAILED status"):
        repository.retry(
            "run-1",
            "step-1",
            updated_at=now,
            started_at=now,
        )


def test_in_memory_repository_retry_missing_step_returns_none():
    repository = InMemoryAgentRunStepsRepository()

    result = repository.retry(
        "run-1",
        "missing",
        updated_at=datetime.now(timezone.utc),
    )

    assert result is None
