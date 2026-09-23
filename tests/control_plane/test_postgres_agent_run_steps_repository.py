from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.control_plane.agent_run_steps.exceptions import (
    DuplicateAgentRunStepError,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_steps.postgres_repository import (
    PostgreSQLAgentRunStepsRepository,
)
from app.control_plane.persistence.models import Base


@pytest.fixture()
def repository():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    session = session_factory()

    try:
        yield PostgreSQLAgentRunStepsRepository(session)
    finally:
        session.close()
        engine.dispose()


def make_step(
    *,
    run_id: str = "run-1",
    step_id: str = "step-1",
    step_index: int = 0,
    step_type: str = "tool",
    status: AgentRunStepStatus = AgentRunStepStatus.PLANNED,
    attempt: int = 1,
    tool_name: str | None = "vehicle_query",
    call_id: str | None = "call-1",
    input=None,
    output=None,
    error: str | None = None,
    failure_category: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    metadata: dict | None = None,
) -> AgentRunStep:
    return AgentRunStep(
        run_id=run_id,
        step_id=step_id,
        step_index=step_index,
        step_type=step_type,
        status=status,
        attempt=attempt,
        tool_name=tool_name,
        call_id=call_id,
        input=input,
        output=output,
        error=error,
        failure_category=failure_category,
        started_at=started_at,
        completed_at=completed_at,
        metadata={} if metadata is None else metadata,
    )


def test_schema_contains_agent_run_steps_table(repository) -> None:
    assert inspect(repository._session.bind).has_table("agent_run_steps")


def test_create_and_get_round_trip(repository) -> None:
    created_at = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)

    step = make_step(
        input={"fleet_id": "fleet-42", "limit": 10},
        metadata={"source": "test", "trace_id": "trace-1"},
        started_at=created_at,
    )

    result = repository.create(step)
    restored = repository.get(step.run_id, step.step_id)

    assert result.run_id == step.run_id
    assert result.step_id == step.step_id
    assert result.step_index == step.step_index
    assert result.step_type == step.step_type
    assert result.status is AgentRunStepStatus.PLANNED
    assert result.attempt == 1
    assert result.tool_name == "vehicle_query"
    assert result.call_id == "call-1"
    assert result.input == {"fleet_id": "fleet-42", "limit": 10}
    assert result.metadata == {"source": "test", "trace_id": "trace-1"}

    assert restored is not None
    assert restored.run_id == result.run_id
    assert restored.step_id == result.step_id
    assert restored.step_index == result.step_index
    assert restored.step_type == result.step_type
    assert restored.status is result.status
    assert restored.attempt == result.attempt
    assert restored.tool_name == result.tool_name
    assert restored.call_id == result.call_id
    assert restored.input == result.input
    assert restored.output == result.output
    assert restored.error == result.error
    assert restored.failure_category == result.failure_category
    assert restored.metadata == result.metadata
    assert restored.started_at == result.started_at.replace(tzinfo=None)
    assert restored.completed_at == result.completed_at


def test_create_rejects_duplicate_step(repository) -> None:
    repository.create(make_step())

    with pytest.raises(
        DuplicateAgentRunStepError,
        match="agent run step already exists: run-1/step-1",
    ):
        repository.create(make_step())


def test_get_missing_step_returns_none(repository) -> None:
    assert repository.get("missing-run", "missing-step") is None


def test_list_orders_by_step_index_then_step_id(repository) -> None:
    repository.create(
        make_step(
            step_id="step-b",
            step_index=2,
        )
    )
    repository.create(
        make_step(
            step_id="step-a",
            step_index=1,
        )
    )
    repository.create(
        make_step(
            step_id="step-c",
            step_index=2,
        )
    )

    steps = repository.list("run-1")

    assert [(step.step_index, step.step_id) for step in steps] == [
        (1, "step-a"),
        (2, "step-b"),
        (2, "step-c"),
    ]


def test_list_filters_by_status(repository) -> None:
    repository.create(
        make_step(
            step_id="step-planned",
            status=AgentRunStepStatus.PLANNED,
        )
    )
    repository.create(
        make_step(
            step_id="step-completed",
            status=AgentRunStepStatus.PLANNED,
        )
    )

    repository.transition(
        "run-1",
        "step-completed",
        status=AgentRunStepStatus.RUNNING,
        updated_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
        started_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
    )
    repository.transition(
        "run-1",
        "step-completed",
        status=AgentRunStepStatus.COMPLETED,
        updated_at=datetime(2026, 9, 23, 10, 2, tzinfo=UTC),
        completed_at=datetime(2026, 9, 23, 10, 2, tzinfo=UTC),
        output={"rows": 4},
    )

    completed = repository.list(
        "run-1",
        status=AgentRunStepStatus.COMPLETED,
    )

    assert [step.step_id for step in completed] == ["step-completed"]


def test_transition_planned_to_running(repository) -> None:
    repository.create(make_step())

    started_at = datetime(2026, 9, 23, 10, 1, tzinfo=UTC)

    updated = repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=started_at,
        started_at=started_at,
    )

    assert updated is not None
    assert updated.status is AgentRunStepStatus.RUNNING
    assert updated.started_at == started_at
    assert updated.completed_at is None


def test_transition_running_to_completed_persists_output(repository) -> None:
    repository.create(make_step())

    started_at = datetime(2026, 9, 23, 10, 1, tzinfo=UTC)
    completed_at = datetime(2026, 9, 23, 10, 2, tzinfo=UTC)

    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=started_at,
        started_at=started_at,
    )

    updated = repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.COMPLETED,
        updated_at=completed_at,
        completed_at=completed_at,
        output={"rows": 4, "source": "delta"},
    )

    assert updated is not None
    assert updated.status is AgentRunStepStatus.COMPLETED
    assert updated.started_at == started_at.replace(tzinfo=None)
    assert updated.completed_at == completed_at
    assert updated.output == {"rows": 4, "source": "delta"}

    restored = repository.get("run-1", "step-1")
    assert restored is not None
    assert restored.output == {"rows": 4, "source": "delta"}
    assert restored.completed_at == completed_at.replace(tzinfo=None)


def test_transition_running_to_failed_persists_error(repository) -> None:
    repository.create(make_step())

    started_at = datetime(2026, 9, 23, 10, 1, tzinfo=UTC)
    failed_at = datetime(2026, 9, 23, 10, 2, tzinfo=UTC)

    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=started_at,
        started_at=started_at,
    )

    updated = repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.FAILED,
        updated_at=failed_at,
        completed_at=failed_at,
        error="vehicle query timed out",
        failure_category="timeout",
    )

    assert updated is not None
    assert updated.status is AgentRunStepStatus.FAILED
    assert updated.error == "vehicle query timed out"
    assert updated.failure_category == "timeout"
    assert updated.completed_at == failed_at


def test_transition_rejects_invalid_lifecycle(repository) -> None:
    repository.create(make_step())

    from app.control_plane.agent_run_steps.exceptions import (
        InvalidAgentRunStepTransitionError,
    )

    with pytest.raises(
        InvalidAgentRunStepTransitionError,
        match="invalid agent run step transition",
    ):
        repository.transition(
            "run-1",
            "step-1",
            status=AgentRunStepStatus.COMPLETED,
            updated_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
            completed_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
        )


def test_transition_requires_completed_at_for_terminal_status(repository) -> None:
    repository.create(make_step())

    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
        started_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
    )

    with pytest.raises(
        ValueError,
        match="completed_at is required for terminal agent run step status",
    ):
        repository.transition(
            "run-1",
            "step-1",
            status=AgentRunStepStatus.COMPLETED,
            updated_at=datetime(2026, 9, 23, 10, 2, tzinfo=UTC),
        )


def test_retry_failed_step_increments_attempt_and_clears_previous_result(
    repository,
) -> None:
    repository.create(make_step())

    started_at = datetime(2026, 9, 23, 10, 1, tzinfo=UTC)
    failed_at = datetime(2026, 9, 23, 10, 2, tzinfo=UTC)
    retry_at = datetime(2026, 9, 23, 10, 3, tzinfo=UTC)

    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=started_at,
        started_at=started_at,
    )
    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.FAILED,
        updated_at=failed_at,
        completed_at=failed_at,
        output={"partial": True},
        error="temporary failure",
        failure_category="transient",
    )

    retried = repository.retry(
        "run-1",
        "step-1",
        updated_at=retry_at,
        started_at=retry_at,
    )

    assert retried is not None
    assert retried.status is AgentRunStepStatus.RUNNING
    assert retried.attempt == 2
    assert retried.started_at == retry_at
    assert retried.completed_at is None
    assert retried.output is None
    assert retried.error is None
    assert retried.failure_category is None


def test_bind_execution_persists_tool_identity_and_input(repository) -> None:
    repository.create(
        make_step(
            run_id="run-bind",
            step_id="retrieve_evidence",
            step_index=0,
            step_type="tool",
            tool_name=None,
            call_id=None,
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

    restored = repository.get("run-bind", "retrieve_evidence")

    assert restored is not None
    assert restored.tool_name == "rag.search"
    assert restored.call_id == "call-123"
    assert restored.input == {"query": "RAG"}


def test_bind_execution_rejects_different_existing_binding(repository) -> None:
    repository.create(
        make_step(
            run_id="run-bind",
            step_id="retrieve_evidence",
            step_index=0,
            step_type="tool",
            tool_name=None,
            call_id=None,
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


def test_bind_execution_rejects_different_input(repository) -> None:
    repository.create(
        make_step(
            run_id="run-bind",
            step_id="retrieve_evidence",
            step_index=0,
            step_type="tool",
            tool_name=None,
            call_id=None,
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

    with pytest.raises(
        ValueError,
        match="agent run step input is already bound to a different value",
    ):
        repository.bind_execution(
            "run-bind",
            "retrieve_evidence",
            tool_name="rag.search",
            call_id="call-123",
            input={"query": "different query"},
        )

    restored = repository.get("run-bind", "retrieve_evidence")

    assert restored is not None
    assert restored.input == {"query": "RAG"}


def test_bind_execution_requires_running_status(repository) -> None:
    repository.create(
        make_step(
            run_id="run-bind",
            step_id="retrieve_evidence",
            step_index=0,
            step_type="tool",
            tool_name=None,
            call_id=None,
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


def test_retry_requires_failed_status(repository) -> None:
    repository.create(make_step())

    with pytest.raises(
        ValueError,
        match="agent run step retry requires FAILED status: planned",
    ):
        repository.retry(
            "run-1",
            "step-1",
            updated_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
        )


def test_transition_missing_step_returns_none(repository) -> None:
    updated = repository.transition(
        "missing-run",
        "missing-step",
        status=AgentRunStepStatus.RUNNING,
        updated_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
    )

    assert updated is None


def test_retry_missing_step_returns_none(repository) -> None:
    retried = repository.retry(
        "missing-run",
        "missing-step",
        updated_at=datetime(2026, 9, 23, 10, 1, tzinfo=UTC),
    )

    assert retried is None


def test_create_can_leave_transaction_uncommitted(repository) -> None:
    step = make_step(step_id="uncommitted-create")

    repository.create(step, commit=False)

    assert repository.get(step.run_id, step.step_id) is not None

    repository._session.rollback()

    assert repository.get(step.run_id, step.step_id) is None


def test_transition_can_leave_transaction_uncommitted(repository) -> None:
    repository.create(make_step())

    started_at = datetime(2026, 9, 23, 10, 1, tzinfo=UTC)

    repository.transition(
        "run-1",
        "step-1",
        status=AgentRunStepStatus.RUNNING,
        updated_at=started_at,
        started_at=started_at,
        commit=False,
    )

    assert repository.get("run-1", "step-1").status is AgentRunStepStatus.RUNNING

    repository._session.rollback()

    restored = repository.get("run-1", "step-1")
    assert restored is not None
    assert restored.status is AgentRunStepStatus.PLANNED
