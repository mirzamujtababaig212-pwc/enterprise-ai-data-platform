from datetime import UTC, datetime

import pytest

from app.control_plane.agent_runs.exceptions import (
    InvalidAgentRunTransitionError,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.repository import AgentRunRepository


def make_run(
    *,
    run_id: str = "run-1",
    agent_name: str = "enterprise-analyst",
    status: AgentRunStatus = AgentRunStatus.PENDING,
    **kwargs,
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        status=status,
        **kwargs,
    )


def test_agent_run_defaults_to_pending() -> None:
    run = make_run()

    assert run.status is AgentRunStatus.PENDING
    assert run.started_at is None
    assert run.completed_at is None
    assert run.error_type is None
    assert run.error_message is None
    assert run.output is None
    assert run.metadata == {}


def test_agent_run_timestamp_is_timezone_aware_when_started() -> None:
    started_at = datetime.now(UTC)

    run = make_run(
        status=AgentRunStatus.RUNNING,
        started_at=started_at,
    )

    assert run.started_at == started_at
    assert run.started_at.tzinfo is not None
    assert run.started_at.utcoffset() is not None


def test_agent_run_accepts_session_and_user_identity() -> None:
    run = make_run(
        session_id="session-123",
        user_id="user-456",
    )

    assert run.session_id == "session-123"
    assert run.user_id == "user-456"


def test_agent_run_allows_valid_lifecycle_transitions() -> None:
    run = make_run()

    running = run.transition_to(AgentRunStatus.RUNNING)
    completed = running.transition_to(AgentRunStatus.COMPLETED)

    assert run.status is AgentRunStatus.PENDING
    assert running.status is AgentRunStatus.RUNNING
    assert completed.status is AgentRunStatus.COMPLETED


def test_agent_run_allows_running_to_failed_transition() -> None:
    run = make_run()

    failed = run.transition_to(AgentRunStatus.RUNNING).transition_to(AgentRunStatus.FAILED)

    assert failed.status is AgentRunStatus.FAILED


def test_agent_run_allows_pending_to_rejected_transition() -> None:
    run = make_run()

    rejected = run.transition_to(AgentRunStatus.REJECTED)

    assert run.status is AgentRunStatus.PENDING
    assert rejected.status is AgentRunStatus.REJECTED


def test_agent_run_allows_waiting_for_approval_to_rejected_transition() -> None:
    run = make_run(status=AgentRunStatus.WAITING_FOR_APPROVAL)

    rejected = run.transition_to(AgentRunStatus.REJECTED)

    assert run.status is AgentRunStatus.WAITING_FOR_APPROVAL
    assert rejected.status is AgentRunStatus.REJECTED


def test_agent_run_allows_failed_to_running_for_recovery() -> None:
    run = make_run(status=AgentRunStatus.FAILED)

    recovered = run.transition_to(AgentRunStatus.RUNNING)

    assert run.status is AgentRunStatus.FAILED
    assert recovered.status is AgentRunStatus.RUNNING


@pytest.mark.parametrize(
    ("current_status", "target_status"),
    [
        (AgentRunStatus.PENDING, AgentRunStatus.COMPLETED),
        (AgentRunStatus.PENDING, AgentRunStatus.FAILED),
        (AgentRunStatus.REJECTED, AgentRunStatus.PENDING),
        (AgentRunStatus.REJECTED, AgentRunStatus.RUNNING),
        (AgentRunStatus.REJECTED, AgentRunStatus.COMPLETED),
        (AgentRunStatus.REJECTED, AgentRunStatus.FAILED),
        (AgentRunStatus.RUNNING, AgentRunStatus.PENDING),
        (AgentRunStatus.COMPLETED, AgentRunStatus.PENDING),
        (AgentRunStatus.COMPLETED, AgentRunStatus.RUNNING),
        (AgentRunStatus.COMPLETED, AgentRunStatus.FAILED),
        (AgentRunStatus.FAILED, AgentRunStatus.PENDING),
        (AgentRunStatus.FAILED, AgentRunStatus.COMPLETED),
    ],
)
def test_agent_run_rejects_invalid_lifecycle_transitions(
    current_status: AgentRunStatus,
    target_status: AgentRunStatus,
) -> None:
    run = make_run(status=current_status)

    with pytest.raises(
        InvalidAgentRunTransitionError,
        match=rf"{current_status.value} -> {target_status.value}",
    ):
        run.transition_to(target_status)


def test_agent_run_transition_does_not_mutate_original() -> None:
    run = make_run()

    transitioned = run.transition_to(AgentRunStatus.RUNNING)

    assert run.status is AgentRunStatus.PENDING
    assert transitioned.status is AgentRunStatus.RUNNING


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.PENDING,
        AgentRunStatus.RUNNING,
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
        AgentRunStatus.REJECTED,
    ],
)
def test_agent_run_supports_all_lifecycle_statuses(status: AgentRunStatus) -> None:
    run = make_run(status=status)

    assert run.status is status


@pytest.mark.parametrize(
    "output",
    [
        "final answer",
        {"answer": "final answer", "sources": ["doc-1"]},
        ["step-1", "step-2"],
        42,
        True,
    ],
)
def test_agent_run_accepts_polymorphic_output(output) -> None:
    run = make_run(
        status=AgentRunStatus.COMPLETED,
        output=output,
        completed_at=datetime.now(UTC),
    )

    assert run.output == output


def test_agent_run_accepts_completion_and_failure_details() -> None:
    completed_at = datetime.now(UTC)

    run = make_run(
        status=AgentRunStatus.FAILED,
        completed_at=completed_at,
        error_type="AgentToolLoopLimitError",
        error_message="maximum tool rounds exceeded",
    )

    assert run.completed_at == completed_at
    assert run.error_type == "AgentToolLoopLimitError"
    assert run.error_message == "maximum tool rounds exceeded"


def test_agent_run_copies_metadata() -> None:
    metadata = {
        "source": "control_plane",
        "request_id": "request-1",
    }

    run = make_run(metadata=metadata)

    assert run.metadata == metadata

    metadata["request_id"] = "changed"

    assert run.metadata["request_id"] == "request-1"


def test_agent_run_rejects_empty_run_id() -> None:
    with pytest.raises(ValueError):
        make_run(run_id="")


def test_agent_run_rejects_empty_agent_name() -> None:
    with pytest.raises(ValueError):
        make_run(agent_name="")


def test_agent_run_repository_protocol_is_importable() -> None:
    assert AgentRunRepository is not None


def test_agent_run_accepts_root_run_hierarchy() -> None:
    run = make_run(
        run_id="root-1",
        root_run_id="root-1",
    )

    assert run.root_run_id == "root-1"
    assert run.parent_run_id is None
    assert run.parent_step_id is None
    assert run.causation_id is None


def test_agent_run_accepts_child_hierarchy() -> None:
    run = make_run(
        run_id="child-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        parent_step_id="delegate-1",
        causation_id="cause-1",
    )

    assert run.root_run_id == "root-1"
    assert run.parent_run_id == "parent-1"
    assert run.parent_step_id == "delegate-1"
    assert run.causation_id == "cause-1"


def test_agent_run_rejects_root_run_with_different_root_id() -> None:
    with pytest.raises(
        ValueError,
        match="root agent runs must use their own run_id",
    ):
        make_run(
            run_id="run-1",
            root_run_id="different-root",
        )


def test_agent_run_rejects_root_run_with_parent_step() -> None:
    with pytest.raises(
        ValueError,
        match="root agent runs cannot specify parent_step_id",
    ):
        make_run(
            run_id="run-1",
            root_run_id="run-1",
            parent_step_id="delegate-1",
        )


def test_agent_run_rejects_child_without_parent_step() -> None:
    with pytest.raises(
        ValueError,
        match="child agent runs must specify parent_step_id",
    ):
        make_run(
            run_id="child-1",
            root_run_id="root-1",
            parent_run_id="parent-1",
        )
