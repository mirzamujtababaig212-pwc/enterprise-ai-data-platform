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


def test_claim_for_recovery_assigns_lease() -> None:
    repository = InMemoryAgentRunRepository()
    failed = make_run("run-recovery", status=AgentRunStatus.FAILED)
    repository.create(failed)

    started_at = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    lease_expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)

    claimed = repository.claim_for_recovery(
        failed.run_id,
        started_at=started_at,
        lease_id="lease-1",
        lease_expires_at=lease_expires_at,
    )

    assert claimed is not None
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.lease_id == "lease-1"
    assert claimed.lease_expires_at == lease_expires_at


def test_heartbeat_extends_matching_lease() -> None:
    repository = InMemoryAgentRunRepository()
    expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)

    run = make_run("run-heartbeat").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-1",
            "lease_expires_at": expires_at,
        }
    )
    repository.create(run)

    new_expiry = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    updated = repository.heartbeat(
        run.run_id,
        lease_id="lease-1",
        lease_expires_at=new_expiry,
    )

    assert updated is not None
    assert updated.lease_id == "lease-1"
    assert updated.lease_expires_at == new_expiry


def test_heartbeat_rejects_wrong_lease() -> None:
    repository = InMemoryAgentRunRepository()
    expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)

    run = make_run("run-heartbeat").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-1",
            "lease_expires_at": expires_at,
        }
    )
    repository.create(run)

    result = repository.heartbeat(
        run.run_id,
        lease_id="lease-wrong",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_heartbeat_rejects_non_running_run() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run("run-completed", status=AgentRunStatus.COMPLETED).model_copy(
        update={
            "lease_id": "lease-1",
            "lease_expires_at": datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        }
    )
    repository.create(run)

    result = repository.heartbeat(
        run.run_id,
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_unexpired_running_run_cannot_be_reclaimed() -> None:
    repository = InMemoryAgentRunRepository()

    lease_expires_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)
    run = make_run("run-stale").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "old-lease",
            "lease_expires_at": lease_expires_at,
        }
    )
    repository.create(run)

    claimed = repository.claim_expired_running_run(
        run.run_id,
        stale_before=datetime(2026, 9, 19, 12, 4, tzinfo=UTC),
        started_at=datetime(2026, 9, 19, 12, 4, tzinfo=UTC),
        lease_id="new-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
    )

    assert claimed is None
    assert repository.get(run.run_id) == run


def test_expired_running_run_can_be_reclaimed() -> None:
    repository = InMemoryAgentRunRepository()

    old_expiry = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    run = make_run("run-stale").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "old-lease",
            "lease_expires_at": old_expiry,
            "error_type": "RuntimeError",
            "error_message": "old failure",
            "output": {"partial": True},
        }
    )
    repository.create(run)

    new_started_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)
    new_expiry = datetime(2026, 9, 19, 12, 6, tzinfo=UTC)

    claimed = repository.claim_expired_running_run(
        run.run_id,
        stale_before=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        started_at=new_started_at,
        lease_id="new-lease",
        lease_expires_at=new_expiry,
    )

    assert claimed is not None
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.started_at == new_started_at
    assert claimed.lease_id == "new-lease"
    assert claimed.lease_expires_at == new_expiry
    assert claimed.completed_at is None
    assert claimed.error_type is None
    assert claimed.error_message is None
    assert claimed.output is None
