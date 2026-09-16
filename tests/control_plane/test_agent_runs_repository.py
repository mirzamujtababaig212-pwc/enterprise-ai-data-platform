from datetime import UTC, datetime

import pytest

from app.control_plane.agent_runs.exceptions import (
    AgentRunNotFoundError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.in_memory import InMemoryAgentRunRepository
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.repository import AgentRunRepository


def make_run(
    run_id: str,
    *,
    agent_name: str = "enterprise-analyst",
    session_id: str | None = None,
    user_id: str | None = None,
    status: AgentRunStatus = AgentRunStatus.PENDING,
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        session_id=session_id,
        user_id=user_id,
        status=status,
    )


def test_repository_implements_agent_run_repository_contract() -> None:
    repository: AgentRunRepository = InMemoryAgentRunRepository()

    assert repository is not None


def test_create_and_get() -> None:
    repository = InMemoryAgentRunRepository()
    run = make_run("run-1")

    created = repository.create(run)

    assert created == run
    assert repository.get("run-1") == run


def test_get_missing_run_returns_none() -> None:
    repository = InMemoryAgentRunRepository()

    assert repository.get("missing") is None


def test_duplicate_create_raises() -> None:
    repository = InMemoryAgentRunRepository()
    run = make_run("run-1")

    repository.create(run)

    with pytest.raises(DuplicateAgentRunError, match="run-1"):
        repository.create(run)


def test_update_existing_run() -> None:
    repository = InMemoryAgentRunRepository()
    run = make_run("run-1")
    repository.create(run)

    started_at = datetime.now(UTC)
    updated = run.model_copy(
        update={
            "status": AgentRunStatus.COMPLETED,
            "started_at": started_at,
            "completed_at": datetime.now(UTC),
            "output": {"answer": "completed"},
        }
    )

    result = repository.update(updated)

    assert result == updated
    assert repository.get("run-1") == updated


def test_update_missing_run_raises() -> None:
    repository = InMemoryAgentRunRepository()

    with pytest.raises(AgentRunNotFoundError, match="missing"):
        repository.update(make_run("missing"))


def test_list_returns_recent_runs() -> None:
    repository = InMemoryAgentRunRepository()

    for run_id in ("run-1", "run-2", "run-3"):
        repository.create(make_run(run_id))

    assert [run.run_id for run in repository.list()] == [
        "run-1",
        "run-2",
        "run-3",
    ]


def test_list_filters_by_agent_name() -> None:
    repository = InMemoryAgentRunRepository()

    repository.create(make_run("run-1", agent_name="analyst"))
    repository.create(make_run("run-2", agent_name="writer"))
    repository.create(make_run("run-3", agent_name="analyst"))

    assert [run.run_id for run in repository.list(agent_name="analyst")] == [
        "run-1",
        "run-3",
    ]


def test_list_filters_by_session_id() -> None:
    repository = InMemoryAgentRunRepository()

    repository.create(make_run("run-1", session_id="session-a"))
    repository.create(make_run("run-2", session_id="session-b"))
    repository.create(make_run("run-3", session_id="session-a"))

    assert [run.run_id for run in repository.list(session_id="session-a")] == [
        "run-1",
        "run-3",
    ]


def test_list_filters_by_user_id() -> None:
    repository = InMemoryAgentRunRepository()

    repository.create(make_run("run-1", user_id="user-a"))
    repository.create(make_run("run-2", user_id="user-b"))
    repository.create(make_run("run-3", user_id="user-a"))

    assert [run.run_id for run in repository.list(user_id="user-a")] == [
        "run-1",
        "run-3",
    ]


def test_list_filters_by_status() -> None:
    repository = InMemoryAgentRunRepository()

    repository.create(make_run("run-1", status=AgentRunStatus.PENDING))
    repository.create(make_run("run-2", status=AgentRunStatus.COMPLETED))
    repository.create(make_run("run-3", status=AgentRunStatus.FAILED))

    assert [run.run_id for run in repository.list(status=AgentRunStatus.COMPLETED)] == ["run-2"]


def test_list_applies_limit_to_latest_runs() -> None:
    repository = InMemoryAgentRunRepository()

    for run_id in ("run-1", "run-2", "run-3", "run-4"):
        repository.create(make_run(run_id))

    assert [run.run_id for run in repository.list(limit=2)] == [
        "run-3",
        "run-4",
    ]


def test_clear_removes_all_runs() -> None:
    repository = InMemoryAgentRunRepository()
    repository.create(make_run("run-1"))

    repository.clear()

    assert repository.get("run-1") is None
    assert repository.list() == []
