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
    tenant_id: str | None = "tenant-acme",
    session_id: str | None = None,
    user_id: str | None = None,
    idempotency_key: str | None = None,
    status: AgentRunStatus = AgentRunStatus.PENDING,
    started_at: datetime | None = None,
    lease_id: str | None = None,
    lease_expires_at: datetime | None = None,
    recovery_attempts: int = 0,
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        tenant_id=tenant_id,
        session_id=session_id,
        user_id=user_id,
        idempotency_key=idempotency_key,
        status=status,
        started_at=started_at,
        lease_id=lease_id,
        lease_expires_at=lease_expires_at,
        recovery_attempts=recovery_attempts,
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


def test_get_by_idempotency_key_returns_matching_user_run() -> None:
    repository = InMemoryAgentRunRepository()
    run = make_run(
        "run-1",
        user_id="user-a",
        idempotency_key="request-123",
    )

    repository.create(run)

    assert (
        repository.get_by_idempotency_key(
            "tenant-acme",
            "user-a",
            "request-123",
        )
        == run
    )


def test_get_by_idempotency_key_is_scoped_to_user() -> None:
    repository = InMemoryAgentRunRepository()
    run = make_run(
        "run-1",
        user_id="user-a",
        idempotency_key="request-123",
    )

    repository.create(run)

    assert (
        repository.get_by_idempotency_key(
            "tenant-acme",
            "user-b",
            "request-123",
        )
        is None
    )


def test_get_by_idempotency_key_returns_none_for_missing_key() -> None:
    repository = InMemoryAgentRunRepository()
    repository.create(
        make_run(
            "run-1",
            user_id="user-a",
            idempotency_key="request-123",
        )
    )

    assert (
        repository.get_by_idempotency_key(
            "tenant-acme",
            "user-a",
            "missing-key",
        )
        is None
    )


def test_multiple_null_idempotency_keys_are_allowed() -> None:
    repository = InMemoryAgentRunRepository()

    first = make_run("run-1", user_id="user-a")
    second = make_run("run-2", user_id="user-a")

    repository.create(first)
    repository.create(second)

    assert (
        repository.get_by_idempotency_key(
            "other-tenant",
            "user-a",
            "request-123",
        )
        is None
    )
    assert repository.get("run-1") == first
    assert repository.get("run-2") == second


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


def test_list_expired_running_runs_filters_before_limit() -> None:
    repository = InMemoryAgentRunRepository()

    stale_before = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

    repository.create(
        make_run(
            "unexpired-1",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        )
    )
    repository.create(
        make_run(
            "unexpired-2",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(2026, 9, 19, 12, 6, tzinfo=UTC),
        )
    )
    repository.create(
        make_run(
            "stale-1",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(2026, 9, 19, 11, 58, tzinfo=UTC),
        )
    )
    repository.create(
        make_run(
            "stale-2",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(2026, 9, 19, 11, 59, tzinfo=UTC),
        )
    )
    repository.create(
        make_run(
            "completed",
            status=AgentRunStatus.COMPLETED,
            lease_expires_at=datetime(2026, 9, 19, 13, 0, tzinfo=UTC),
        )
    )

    runs = repository.list_expired_running_runs(
        stale_before=stale_before,
        limit=1,
    )

    assert [run.run_id for run in runs] == ["stale-1"]


def test_list_expired_running_runs_ignores_runs_without_expiry() -> None:
    repository = InMemoryAgentRunRepository()

    stale_before = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

    repository.create(
        make_run(
            "no-expiry",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=None,
        )
    )
    repository.create(
        make_run(
            "stale",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(2026, 9, 19, 11, 59, tzinfo=UTC),
        )
    )

    runs = repository.list_expired_running_runs(
        stale_before=stale_before,
    )

    assert [run.run_id for run in runs] == ["stale"]


def test_clear_removes_all_runs() -> None:
    repository = InMemoryAgentRunRepository()
    repository.create(make_run("run-1"))

    repository.clear()

    assert repository.get("run-1") is None
    assert repository.list() == []


def test_claim_for_recovery_assigns_lease() -> None:
    repository = InMemoryAgentRunRepository()
    failed = make_run(
        "run-recovery",
        status=AgentRunStatus.FAILED,
        started_at=datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
    )
    repository.create(failed)

    started_at = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    lease_expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)

    claimed = repository.claim_for_recovery(
        failed.run_id,
        started_at=started_at,
        lease_id="lease-1",
        lease_expires_at=lease_expires_at,
        max_recovery_attempts=3,
    )

    assert claimed is not None
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.started_at == failed.started_at
    assert claimed.recovery_attempts == failed.recovery_attempts + 1
    assert claimed.lease_id is not None
    assert claimed.lease_expires_at is not None
    assert claimed.lease_expires_at > claimed.started_at
    assert claimed.lease_id == "lease-1"
    assert claimed.lease_expires_at == lease_expires_at


def test_claim_waiting_for_approval_assigns_fresh_lease() -> None:
    repository = InMemoryAgentRunRepository()
    waiting = make_run(
        "run-approval-continuation",
        status=AgentRunStatus.WAITING_FOR_APPROVAL,
        recovery_attempts=2,
    )
    repository.create(waiting)

    lease_expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)

    claimed = repository.claim_waiting_for_approval(
        waiting.run_id,
        lease_id="approval-lease-1",
        lease_expires_at=lease_expires_at,
    )

    assert claimed is not None
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.started_at == waiting.started_at
    assert claimed.recovery_attempts == waiting.recovery_attempts
    assert claimed.lease_id == "approval-lease-1"
    assert claimed.lease_expires_at == lease_expires_at
    assert claimed.completed_at is None
    assert claimed.error_type is None
    assert claimed.error_message is None


def test_claim_waiting_for_approval_rejects_non_waiting_run() -> None:
    repository = InMemoryAgentRunRepository()
    run = make_run(
        "run-approval-wrong-status",
        status=AgentRunStatus.RUNNING,
    )
    repository.create(run)

    claimed = repository.claim_waiting_for_approval(
        run.run_id,
        lease_id="approval-lease-2",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
    )

    assert claimed is None
    assert repository.get(run.run_id) == run


def test_claim_waiting_for_approval_rejects_cancelled_run() -> None:
    repository = InMemoryAgentRunRepository()
    waiting = make_run(
        "run-approval-cancelled",
        status=AgentRunStatus.WAITING_FOR_APPROVAL,
    ).model_copy(update={"cancellation_requested": True})
    repository.create(waiting)

    claimed = repository.claim_waiting_for_approval(
        waiting.run_id,
        lease_id="approval-lease-3",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
    )

    assert claimed is None
    assert repository.get(waiting.run_id) == waiting


def test_claim_waiting_for_approval_is_single_use() -> None:
    repository = InMemoryAgentRunRepository()
    waiting = make_run(
        "run-approval-single-use",
        status=AgentRunStatus.WAITING_FOR_APPROVAL,
    )
    repository.create(waiting)

    first = repository.claim_waiting_for_approval(
        waiting.run_id,
        lease_id="approval-lease-4",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
    )
    second = repository.claim_waiting_for_approval(
        waiting.run_id,
        lease_id="approval-lease-5",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert first is not None
    assert second is None

    restored = repository.get(waiting.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "approval-lease-4"
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 1, tzinfo=UTC)


def test_reject_waiting_for_approval_marks_run_rejected() -> None:
    repository = InMemoryAgentRunRepository()
    waiting = make_run(
        "run-approval-rejected",
        status=AgentRunStatus.WAITING_FOR_APPROVAL,
    )
    repository.create(waiting)

    completed_at = datetime(2026, 9, 19, 12, 3, tzinfo=UTC)

    rejected = repository.reject_waiting_for_approval(
        waiting.run_id,
        completed_at=completed_at,
        error_type="ApprovalRejected",
        error_message="Payment approval was rejected.",
    )

    assert rejected is not None
    assert rejected.status is AgentRunStatus.REJECTED
    assert rejected.completed_at == completed_at
    assert rejected.error_type == "ApprovalRejected"
    assert rejected.error_message == "Payment approval was rejected."
    assert rejected.lease_id is None
    assert rejected.lease_expires_at is None
    assert repository.get(waiting.run_id) == rejected


def test_reject_waiting_for_approval_rejects_non_waiting_run() -> None:
    repository = InMemoryAgentRunRepository()
    run = make_run(
        "run-approval-reject-wrong-status",
        status=AgentRunStatus.RUNNING,
    )
    repository.create(run)

    result = repository.reject_waiting_for_approval(
        run.run_id,
        completed_at=datetime(2026, 9, 19, 12, 3, tzinfo=UTC),
        error_type="ApprovalRejected",
        error_message="Payment approval was rejected.",
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_reject_waiting_for_approval_rejects_cancelled_run() -> None:
    repository = InMemoryAgentRunRepository()
    waiting = make_run(
        "run-approval-reject-cancelled",
        status=AgentRunStatus.WAITING_FOR_APPROVAL,
    ).model_copy(update={"cancellation_requested": True})
    repository.create(waiting)

    result = repository.reject_waiting_for_approval(
        waiting.run_id,
        completed_at=datetime(2026, 9, 19, 12, 3, tzinfo=UTC),
        error_type="ApprovalRejected",
        error_message="Payment approval was rejected.",
    )

    assert result is None
    assert repository.get(waiting.run_id) == waiting


def test_claim_for_recovery_rejects_exhausted_run() -> None:
    repository = InMemoryAgentRunRepository()
    failed = make_run(
        "run-recovery-exhausted",
        status=AgentRunStatus.FAILED,
        recovery_attempts=3,
    )
    repository.create(failed)

    claimed = repository.claim_for_recovery(
        failed.run_id,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        max_recovery_attempts=3,
    )

    assert claimed is None
    assert repository.get(failed.run_id) == failed


def test_claim_expired_running_run_rejects_exhausted_run() -> None:
    repository = InMemoryAgentRunRepository()
    expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    run = make_run(
        "run-stale-exhausted",
        status=AgentRunStatus.RUNNING,
        recovery_attempts=3,
        lease_id="old-lease",
        lease_expires_at=expires_at,
    )
    repository.create(run)

    claimed = repository.claim_expired_running_run(
        run.run_id,
        stale_before=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        started_at=datetime(2026, 9, 19, 12, 3, tzinfo=UTC),
        lease_id="new-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 4, tzinfo=UTC),
        max_recovery_attempts=3,
    )

    assert claimed is None
    assert repository.get(run.run_id) == run


def test_fail_recovery_exhausted_transitions_stale_run_to_failed() -> None:
    repository = InMemoryAgentRunRepository()
    expired_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    run = make_run(
        "run-fail-recovery-exhausted",
        status=AgentRunStatus.RUNNING,
        recovery_attempts=3,
        lease_id="expired-lease",
        lease_expires_at=expired_at,
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    failed = repository.fail_recovery_exhausted(
        run.run_id,
        completed_at=completed_at,
        max_recovery_attempts=3,
        error_type="RecoveryExhaustedError",
        error_message="maximum recovery attempts exceeded",
    )

    assert failed is not None
    assert failed.status is AgentRunStatus.FAILED
    assert failed.completed_at == completed_at
    assert failed.error_type == "RecoveryExhaustedError"
    assert failed.error_message == "maximum recovery attempts exceeded"
    assert failed.recovery_attempts == 3
    assert failed.lease_id is None
    assert failed.lease_expires_at is None

    restored = repository.get(run.run_id)
    assert restored == failed

    second_attempt = repository.fail_recovery_exhausted(
        run.run_id,
        completed_at=datetime(2026, 9, 19, 12, 3, tzinfo=UTC),
        max_recovery_attempts=3,
        error_type="RecoveryExhaustedError",
        error_message="duplicate exhaustion transition",
    )

    assert second_attempt is None
    assert repository.get(run.run_id) == failed


def test_fail_recovery_exhausted_rejects_active_lease() -> None:
    repository = InMemoryAgentRunRepository()
    run = make_run(
        "run-active-recovery",
        status=AgentRunStatus.RUNNING,
        recovery_attempts=3,
        lease_id="active-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
    )
    repository.create(run)

    result = repository.fail_recovery_exhausted(
        run.run_id,
        completed_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        max_recovery_attempts=3,
        error_type="RecoveryExhaustedError",
        error_message="should not transition while lease is active",
    )

    assert result is None
    assert repository.get(run.run_id) == run


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
        max_recovery_attempts=3,
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
        max_recovery_attempts=3,
    )

    assert claimed is not None
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.started_at == run.started_at
    assert claimed.recovery_attempts == run.recovery_attempts + 1
    assert claimed.lease_id == "new-lease"
    assert claimed.lease_expires_at == new_expiry
    assert claimed.completed_at is None
    assert claimed.error_type is None
    assert claimed.error_message is None
    assert claimed.output is None


def test_complete_if_owner_clears_lease():
    repository = InMemoryAgentRunRepository()
    run = make_run("run-complete").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        }
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.complete_if_owner(
        run.run_id,
        lease_id="lease-a",
        completed_at=completed_at,
        output={"answer": "done"},
    )

    assert result is not None
    assert result.status is AgentRunStatus.COMPLETED
    assert result.completed_at == completed_at
    assert result.output == {"answer": "done"}
    assert result.lease_id is None
    assert result.lease_expires_at is None


def test_complete_if_owner_rejects_wrong_lease():
    repository = InMemoryAgentRunRepository()
    run = make_run("run-complete-wrong").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        }
    )
    repository.create(run)

    result = repository.complete_if_owner(
        run.run_id,
        lease_id="lease-b",
        completed_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        output={"answer": "stale"},
    )

    assert result is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "lease-a"


def test_fail_if_owner_clears_lease():
    repository = InMemoryAgentRunRepository()
    run = make_run("run-fail").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        }
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.fail_if_owner(
        run.run_id,
        lease_id="lease-a",
        completed_at=completed_at,
        error_type="RuntimeError",
        error_message="agent failed",
    )

    assert result is not None
    assert result.status is AgentRunStatus.FAILED
    assert result.completed_at == completed_at
    assert result.error_type == "RuntimeError"
    assert result.error_message == "agent failed"
    assert result.lease_id is None
    assert result.lease_expires_at is None


def test_fail_if_owner_rejects_wrong_lease():
    repository = InMemoryAgentRunRepository()
    run = make_run("run-fail-wrong").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        }
    )
    repository.create(run)

    result = repository.fail_if_owner(
        run.run_id,
        lease_id="lease-b",
        completed_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        error_type="RuntimeError",
        error_message="stale executor",
    )

    assert result is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "lease-a"


def test_cancel_if_owner_clears_lease() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run("run-cancel").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        }
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.cancel_if_owner(
        run.run_id,
        lease_id="lease-a",
        completed_at=completed_at,
    )

    assert result is not None
    assert result.status is AgentRunStatus.CANCELLED
    assert result.completed_at == completed_at
    assert result.lease_id is None
    assert result.lease_expires_at is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.CANCELLED
    assert restored.lease_id is None
    assert restored.lease_expires_at is None


def test_cancel_if_owner_rejects_wrong_lease() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run("run-cancel-wrong").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        }
    )
    repository.create(run)

    result = repository.cancel_if_owner(
        run.run_id,
        lease_id="lease-b",
        completed_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_complete_if_owner_rejects_expired_lease():
    repository = InMemoryAgentRunRepository()
    run = make_run("run-complete-expired").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        }
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.complete_if_owner(
        run.run_id,
        lease_id="lease-a",
        completed_at=completed_at,
        output={"answer": "late"},
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_complete_if_owner_rejects_exact_expiry():
    repository = InMemoryAgentRunRepository()
    run = make_run("run-complete-exact-expiry").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        }
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.complete_if_owner(
        run.run_id,
        lease_id="lease-a",
        completed_at=completed_at,
        output={"answer": "late"},
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_fail_if_owner_rejects_expired_lease():
    repository = InMemoryAgentRunRepository()
    run = make_run("run-fail-expired").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        }
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.fail_if_owner(
        run.run_id,
        lease_id="lease-a",
        completed_at=completed_at,
        error_type="RuntimeError",
        error_message="late failure",
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_cancel_if_owner_rejects_expired_lease():
    repository = InMemoryAgentRunRepository()
    run = make_run("run-cancel-expired").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        }
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.cancel_if_owner(
        run.run_id,
        lease_id="lease-a",
        completed_at=completed_at,
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_request_cancellation_marks_running_run() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run(
        "run-cancellation-request",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-a",
    )
    repository.create(run)

    requested_at = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

    result = repository.request_cancellation(
        run.run_id,
        requested_at=requested_at,
    )

    assert result is not None
    assert result.cancellation_requested is True
    assert result.cancellation_requested_at == requested_at

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.cancellation_requested is True
    assert restored.cancellation_requested_at == requested_at


def test_request_cancellation_is_idempotent_and_preserves_first_timestamp() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run(
        "run-cancellation-idempotent",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-a",
    )
    repository.create(run)

    first_requested_at = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    second_requested_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)

    first = repository.request_cancellation(
        run.run_id,
        requested_at=first_requested_at,
    )
    second = repository.request_cancellation(
        run.run_id,
        requested_at=second_requested_at,
    )

    assert first is not None
    assert second is not None
    assert first.cancellation_requested is True
    assert second.cancellation_requested is True
    assert first.cancellation_requested_at == first_requested_at
    assert second.cancellation_requested_at == first_requested_at


def test_request_cancellation_returns_none_for_missing_run() -> None:
    repository = InMemoryAgentRunRepository()

    result = repository.request_cancellation(
        "missing",
        requested_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert result is None


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.PENDING,
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
        AgentRunStatus.CANCELLED,
        AgentRunStatus.REJECTED,
    ],
)
def test_request_cancellation_returns_none_for_non_running_run(
    status: AgentRunStatus,
) -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run(
        f"run-cancellation-{status.value}",
        status=status,
    )
    repository.create(run)

    result = repository.request_cancellation(
        run.run_id,
        requested_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert result is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.cancellation_requested is False
    assert restored.cancellation_requested_at is None


def test_cancelled_run_is_terminal() -> None:
    run = make_run("run-cancelled", status=AgentRunStatus.RUNNING)

    cancelled = run.transition_to(AgentRunStatus.CANCELLED)

    assert cancelled.status is AgentRunStatus.CANCELLED

    with pytest.raises(Exception):
        cancelled.transition_to(AgentRunStatus.RUNNING)


def test_transition_to_waiting_for_approval_if_owner_clears_lease() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run("run-approval-wait").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        }
    )
    repository.create(run)

    updated_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.transition_to_waiting_for_approval_if_owner(
        run.run_id,
        lease_id="lease-a",
        updated_at=updated_at,
    )

    assert result is not None
    assert result.status is AgentRunStatus.WAITING_FOR_APPROVAL
    assert result.completed_at is None
    assert result.lease_id is None
    assert result.lease_expires_at is None

    restored = repository.get(run.run_id)
    assert restored == result


def test_transition_to_waiting_for_approval_if_owner_rejects_wrong_lease() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run("run-approval-wait-wrong-lease").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        }
    )
    repository.create(run)

    result = repository.transition_to_waiting_for_approval_if_owner(
        run.run_id,
        lease_id="lease-b",
        updated_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_transition_to_waiting_for_approval_if_owner_rejects_expired_lease() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run("run-approval-wait-expired").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        }
    )
    repository.create(run)

    result = repository.transition_to_waiting_for_approval_if_owner(
        run.run_id,
        lease_id="lease-a",
        updated_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert result is None
    assert repository.get(run.run_id) == run


def test_transition_to_waiting_for_approval_if_owner_rejects_exact_expiry() -> None:
    repository = InMemoryAgentRunRepository()

    run = make_run("run-approval-wait-exact-expiry").model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
            "lease_id": "lease-a",
            "lease_expires_at": datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        }
    )
    repository.create(run)

    updated_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.transition_to_waiting_for_approval_if_owner(
        run.run_id,
        lease_id="lease-a",
        updated_at=updated_at,
    )

    assert result is None
    assert repository.get(run.run_id) == run
