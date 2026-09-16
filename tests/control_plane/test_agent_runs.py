from datetime import UTC, datetime

import pytest

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


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.PENDING,
        AgentRunStatus.RUNNING,
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
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
