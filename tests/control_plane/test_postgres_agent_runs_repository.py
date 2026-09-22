from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.control_plane.agent_runs.exceptions import (
    AgentRunNotFoundError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from ai_platform.agents.budget import ExecutionBudget
from ai_platform.agents.models import AgentRequest
from rag.governance.policy import GovernancePolicy
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
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
        yield PostgreSQLAgentRunRepository(session)
    finally:
        session.close()
        engine.dispose()


def make_run(
    *,
    run_id: str = "run-1",
    agent_name: str = "vehicle-agent",
    session_id: str | None = "session-1",
    user_id: str | None = "user-1",
    principal: str | None = None,
    idempotency_key: str | None = None,
    status: AgentRunStatus = AgentRunStatus.PENDING,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    lease_id: str | None = None,
    lease_expires_at: datetime | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    output=None,
    metadata: dict | None = None,
    request_snapshot: AgentRunRequestSnapshot | None = None,
    recovery_attempts: int = 0,
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        session_id=session_id,
        user_id=user_id,
        principal=principal,
        idempotency_key=idempotency_key,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        lease_id=lease_id,
        lease_expires_at=lease_expires_at,
        error_type=error_type,
        error_message=error_message,
        output=output,
        metadata={} if metadata is None else metadata,
        request_snapshot=request_snapshot,
        recovery_attempts=recovery_attempts,
    )


def test_schema_contains_agent_runs_table(repository) -> None:
    assert inspect(repository._session.bind).has_table("agent_runs")


def test_create_and_get_round_trip(repository) -> None:
    started_at = datetime(2026, 9, 17, 10, 0, tzinfo=UTC)

    run = make_run(
        started_at=started_at,
        status=AgentRunStatus.RUNNING,
        output={"step": "started"},
        metadata={"source": "test", "attempt": 1},
        principal="api_key:test-principal",
    )

    result = repository.create(run)
    restored = repository.get(run.run_id)

    assert result == run
    assert restored is not None
    assert restored.run_id == run.run_id
    assert restored.agent_name == run.agent_name
    assert restored.session_id == run.session_id
    assert restored.user_id == run.user_id
    assert restored.principal == run.principal
    assert restored.status == AgentRunStatus.RUNNING
    assert restored.started_at == started_at
    assert restored.output == {"step": "started"}
    assert restored.metadata == {"source": "test", "attempt": 1}


def test_request_snapshot_round_trip_reconstructs_request(repository) -> None:
    request = AgentRequest(
        input="Find vehicle incidents for fleet-42",
        session_id="session-123",
        user_id="user-456",
        principal="api_key:snapshot-principal",
        memory_namespace="fleet-memory",
        governance_policy=GovernancePolicy(
            required_metadata={
                "tenant_id": "tenant-1",
                "classification": "internal",
            }
        ),
        metadata={
            "request_id": "req-789",
            "source": "control_plane",
        },
        execution_budget=ExecutionBudget(
            max_llm_calls=7,
            max_tool_calls=11,
            max_tool_rounds=4,
            max_duration_seconds=42.5,
        ),
    )

    snapshot = AgentRunRequestSnapshot.from_request(request)

    run = make_run(
        run_id="snapshot-round-trip",
        session_id=request.session_id,
        user_id=request.user_id,
        principal=request.principal,
        request_snapshot=snapshot,
    )

    repository.create(run)

    restored = repository.get(run.run_id)

    assert restored is not None
    assert restored.request_snapshot is not None
    assert restored.request_snapshot.schema_version == 1

    reconstructed = restored.request_snapshot.to_request(
        session_id=restored.session_id,
        user_id=restored.user_id,
        principal=restored.principal,
    )

    assert reconstructed.input == request.input
    assert reconstructed.session_id == request.session_id
    assert reconstructed.user_id == request.user_id
    assert reconstructed.principal == request.principal
    assert reconstructed.memory_namespace == request.memory_namespace
    assert reconstructed.governance_policy is not None
    assert reconstructed.governance_policy.required_metadata == (
        request.governance_policy.required_metadata
    )
    assert reconstructed.metadata == request.metadata
    assert reconstructed.execution_budget is not None
    assert reconstructed.execution_budget.max_llm_calls == 7
    assert reconstructed.execution_budget.max_tool_calls == 11
    assert reconstructed.execution_budget.max_tool_rounds == 4
    assert reconstructed.execution_budget.max_duration_seconds == 42.5


def test_get_missing_run_returns_none(repository) -> None:
    assert repository.get("does-not-exist") is None


def test_create_can_leave_transaction_uncommitted(repository) -> None:
    run = make_run(run_id="uncommitted-create")

    repository.create(run, commit=False)

    assert repository.get(run.run_id) == run

    repository._session.rollback()

    assert repository.get(run.run_id) is None


def test_update_can_leave_transaction_uncommitted(repository) -> None:
    repository.create(make_run())

    updated = make_run(
        status=AgentRunStatus.COMPLETED,
        output={"answer": "updated"},
    )

    repository.update(updated, commit=False)

    assert repository.get(updated.run_id) == updated

    repository._session.rollback()

    restored = repository.get(updated.run_id)
    assert restored is not None
    assert restored.status == AgentRunStatus.PENDING
    assert restored.output is None


def test_create_rejects_duplicate_run_id(repository) -> None:
    repository.create(make_run())

    with pytest.raises(
        DuplicateAgentRunError,
        match="agent run already exists: run-1",
    ):
        repository.create(make_run())


def test_claim_for_recovery_claims_failed_run(repository) -> None:
    failed = make_run(
        status=AgentRunStatus.FAILED,
        started_at=datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 17, 10, 5, tzinfo=UTC),
        error_type="RuntimeError",
        error_message="previous attempt failed",
        output={"partial": "output"},
        metadata={"attempt": 1},
    )
    repository.create(failed)

    recovery_started_at = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

    claimed = repository.claim_for_recovery(
        failed.run_id,
        started_at=recovery_started_at,
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        max_recovery_attempts=3,
    )

    assert claimed is not None
    assert claimed.run_id == failed.run_id
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.started_at == failed.started_at
    assert claimed.recovery_attempts == failed.recovery_attempts + 1
    assert claimed.completed_at is None
    assert claimed.error_type is None
    assert claimed.error_message is None
    assert claimed.output is None
    assert claimed.metadata == failed.metadata

    restored = repository.get(failed.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.started_at == failed.started_at


def test_claim_for_recovery_rejects_exhausted_run(repository) -> None:
    failed = make_run(
        run_id="pg-recovery-exhausted",
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

    restored = repository.get(failed.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.FAILED
    assert restored.recovery_attempts == 3
    assert restored.lease_id is None


def test_claim_expired_running_run_rejects_exhausted_run(repository) -> None:
    run = make_run(
        run_id="pg-stale-recovery-exhausted",
        status=AgentRunStatus.RUNNING,
        recovery_attempts=3,
        lease_id="old-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.recovery_attempts == 3
    assert restored.lease_id == "old-lease"


def test_fail_recovery_exhausted_transitions_stale_run_to_failed(repository) -> None:
    run = make_run(
        run_id="pg-fail-recovery-exhausted",
        status=AgentRunStatus.RUNNING,
        recovery_attempts=3,
        lease_id="expired-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
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


def test_fail_recovery_exhausted_rejects_active_lease(repository) -> None:
    run = make_run(
        run_id="pg-active-recovery",
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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.recovery_attempts == 3
    assert restored.lease_id == "active-lease"


def test_claim_for_recovery_returns_none_for_non_failed_run(repository) -> None:
    running = make_run(status=AgentRunStatus.RUNNING)
    repository.create(running)

    claimed = repository.claim_for_recovery(
        running.run_id,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        max_recovery_attempts=3,
    )

    assert claimed is None

    restored = repository.get(running.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING


def test_claim_for_recovery_returns_none_for_cancellation_requested_run(
    repository,
) -> None:
    run = make_run(
        status=AgentRunStatus.FAILED,
    ).model_copy(
        update={
            "cancellation_requested": True,
            "cancellation_requested_at": datetime(
                2026,
                9,
                19,
                11,
                58,
                tzinfo=UTC,
            ),
        }
    )
    repository.create(run)

    claimed = repository.claim_for_recovery(
        run.run_id,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="new-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
        max_recovery_attempts=3,
    )

    assert claimed is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.FAILED
    assert restored.cancellation_requested is True
    assert restored.cancellation_requested_at == datetime(
        2026,
        9,
        19,
        11,
        58,
        tzinfo=UTC,
    )


def test_claim_for_recovery_returns_none_for_missing_run(repository) -> None:
    claimed = repository.claim_for_recovery(
        "does-not-exist",
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        max_recovery_attempts=3,
    )

    assert claimed is None


def test_claim_for_recovery_can_only_claim_once(repository) -> None:
    failed = make_run(
        status=AgentRunStatus.FAILED,
        error_type="RuntimeError",
        error_message="previous attempt failed",
    )
    repository.create(failed)

    first_started_at = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    second_started_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)

    first_claim = repository.claim_for_recovery(
        failed.run_id,
        started_at=first_started_at,
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        max_recovery_attempts=3,
    )
    second_claim = repository.claim_for_recovery(
        failed.run_id,
        started_at=second_started_at,
        lease_id="lease-2",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        max_recovery_attempts=3,
    )

    assert first_claim is not None
    assert first_claim.status is AgentRunStatus.RUNNING
    assert first_claim.started_at == failed.started_at
    assert first_claim.recovery_attempts == failed.recovery_attempts + 1

    assert second_claim is None

    restored = repository.get(failed.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.started_at == failed.started_at


def test_update_round_trip(repository) -> None:
    repository.create(make_run())

    completed_at = datetime(2026, 9, 17, 10, 5, tzinfo=UTC)

    updated = make_run(
        status=AgentRunStatus.COMPLETED,
        started_at=datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
        completed_at=completed_at,
        output={"answer": "completed"},
        metadata={"source": "agent_execution"},
    )

    result = repository.update(updated)
    restored = repository.get(updated.run_id)

    assert result == updated
    assert restored is not None
    assert restored.status == AgentRunStatus.COMPLETED
    assert restored.completed_at == completed_at
    assert restored.output == {"answer": "completed"}
    assert restored.metadata == {"source": "agent_execution"}


def test_update_failure_details(repository) -> None:
    repository.create(make_run())

    failed = make_run(
        status=AgentRunStatus.FAILED,
        started_at=datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 17, 10, 1, tzinfo=UTC),
        error_type="RuntimeError",
        error_message="agent failed",
    )

    repository.update(failed)

    restored = repository.get(failed.run_id)

    assert restored is not None
    assert restored.status == AgentRunStatus.FAILED
    assert restored.error_type == "RuntimeError"
    assert restored.error_message == "agent failed"


def test_lifecycle_progression_persists_completed_run(repository) -> None:
    run = make_run()

    repository.create(run)

    running = run.transition_to(AgentRunStatus.RUNNING).model_copy(
        update={
            "started_at": datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
        }
    )
    repository.update(running)

    restored_running = repository.get(run.run_id)
    assert restored_running is not None
    assert restored_running.status == AgentRunStatus.RUNNING
    assert restored_running.started_at == running.started_at

    completed = running.transition_to(AgentRunStatus.COMPLETED).model_copy(
        update={
            "completed_at": datetime(2026, 9, 17, 10, 5, tzinfo=UTC),
            "output": {"answer": "completed"},
        }
    )
    repository.update(completed)

    restored_completed = repository.get(run.run_id)
    assert restored_completed is not None
    assert restored_completed.status == AgentRunStatus.COMPLETED
    assert restored_completed.started_at == running.started_at
    assert restored_completed.completed_at == completed.completed_at
    assert restored_completed.output == {"answer": "completed"}


def test_lifecycle_progression_persists_failed_run(repository) -> None:
    run = make_run()

    repository.create(run)

    running = run.transition_to(AgentRunStatus.RUNNING).model_copy(
        update={
            "started_at": datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
        }
    )
    repository.update(running)

    failed = running.transition_to(AgentRunStatus.FAILED).model_copy(
        update={
            "completed_at": datetime(2026, 9, 17, 10, 1, tzinfo=UTC),
            "error_type": "RuntimeError",
            "error_message": "agent failed",
        }
    )
    repository.update(failed)

    restored_failed = repository.get(run.run_id)
    assert restored_failed is not None
    assert restored_failed.status == AgentRunStatus.FAILED
    assert restored_failed.started_at == running.started_at
    assert restored_failed.completed_at == failed.completed_at
    assert restored_failed.error_type == "RuntimeError"
    assert restored_failed.error_message == "agent failed"


def test_update_missing_run_raises(repository) -> None:
    run = make_run(run_id="missing-run")

    with pytest.raises(
        AgentRunNotFoundError,
        match="agent run not found: missing-run",
    ):
        repository.update(run)


def test_list_returns_runs(repository) -> None:
    repository.create(make_run(run_id="run-1"))
    repository.create(make_run(run_id="run-2"))

    runs = repository.list()

    assert len(runs) == 2
    assert {run.run_id for run in runs} == {"run-1", "run-2"}


def test_list_filters_by_agent_name(repository) -> None:
    repository.create(make_run(run_id="run-1", agent_name="agent-a"))
    repository.create(make_run(run_id="run-2", agent_name="agent-b"))

    runs = repository.list(agent_name="agent-a")

    assert [run.run_id for run in runs] == ["run-1"]


def test_list_filters_by_session_id(repository) -> None:
    repository.create(make_run(run_id="run-1", session_id="session-a"))
    repository.create(make_run(run_id="run-2", session_id="session-b"))

    runs = repository.list(session_id="session-a")

    assert [run.run_id for run in runs] == ["run-1"]


def test_list_filters_by_user_id(repository) -> None:
    repository.create(make_run(run_id="run-1", user_id="user-a"))
    repository.create(make_run(run_id="run-2", user_id="user-b"))

    runs = repository.list(user_id="user-a")

    assert [run.run_id for run in runs] == ["run-1"]


def test_list_filters_by_status(repository) -> None:
    repository.create(
        make_run(
            run_id="run-1",
            status=AgentRunStatus.RUNNING,
        )
    )
    repository.create(
        make_run(
            run_id="run-2",
            status=AgentRunStatus.COMPLETED,
        )
    )

    runs = repository.list(status=AgentRunStatus.COMPLETED)

    assert [run.run_id for run in runs] == ["run-2"]


def test_list_respects_limit(repository) -> None:
    for index in range(5):
        repository.create(make_run(run_id=f"run-{index}"))

    runs = repository.list(limit=2)

    assert len(runs) == 2


def test_repository_implements_contract(repository) -> None:
    from app.control_plane.agent_runs.repository import AgentRunRepository

    contract: AgentRunRepository = repository

    assert contract is not None


def test_claim_for_recovery_persists_lease(repository) -> None:
    failed = make_run(status=AgentRunStatus.FAILED)
    repository.create(failed)

    started_at = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)

    claimed = repository.claim_for_recovery(
        failed.run_id,
        started_at=started_at,
        lease_id="lease-recovery",
        lease_expires_at=expires_at,
        max_recovery_attempts=3,
    )

    assert claimed is not None
    assert claimed.lease_id == "lease-recovery"
    assert claimed.lease_expires_at == expires_at

    restored = repository.get(failed.run_id)
    assert restored is not None
    assert restored.lease_id == "lease-recovery"
    assert restored.lease_expires_at == expires_at


def test_heartbeat_extends_matching_lease(repository) -> None:
    expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    run = make_run(
        status=AgentRunStatus.RUNNING,
        lease_id="lease-1",
        lease_expires_at=expires_at,
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


def test_heartbeat_rejects_wrong_lease(repository) -> None:
    expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    run = make_run(
        status=AgentRunStatus.RUNNING,
        lease_id="lease-1",
        lease_expires_at=expires_at,
    )
    repository.create(run)

    result = repository.heartbeat(
        run.run_id,
        lease_id="wrong-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert result is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.lease_id == "lease-1"
    assert restored.lease_expires_at == expires_at


def test_heartbeat_rejects_non_running_run(repository) -> None:
    run = make_run(
        status=AgentRunStatus.COMPLETED,
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
    )
    repository.create(run)

    result = repository.heartbeat(
        run.run_id,
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert result is None


def test_unexpired_running_run_cannot_be_reclaimed(repository) -> None:
    run = make_run(
        status=AgentRunStatus.RUNNING,
        lease_id="old-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.lease_id == "old-lease"


def test_expired_running_run_returns_none_for_cancellation_requested_run(
    repository,
) -> None:
    run = make_run(
        status=AgentRunStatus.RUNNING,
        lease_id="old-lease",
        lease_expires_at=datetime(
            2026,
            9,
            19,
            11,
            59,
            tzinfo=UTC,
        ),
    ).model_copy(
        update={
            "cancellation_requested": True,
            "cancellation_requested_at": datetime(
                2026,
                9,
                19,
                11,
                58,
                tzinfo=UTC,
            ),
        }
    )
    repository.create(run)

    claimed = repository.claim_expired_running_run(
        run.run_id,
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        started_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        lease_id="new-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        max_recovery_attempts=3,
    )

    assert claimed is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "old-lease"
    assert restored.cancellation_requested is True


def test_expired_running_run_can_be_reclaimed(repository) -> None:
    original_started_at = datetime(2026, 9, 19, 11, 0, tzinfo=UTC)

    run = make_run(
        status=AgentRunStatus.RUNNING,
        started_at=original_started_at,
        lease_id="old-lease",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
        output={"partial": True},
    )
    repository.create(run)

    started_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)
    expires_at = datetime(2026, 9, 19, 12, 6, tzinfo=UTC)

    claimed = repository.claim_expired_running_run(
        run.run_id,
        stale_before=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
        started_at=started_at,
        lease_id="new-lease",
        lease_expires_at=expires_at,
        max_recovery_attempts=3,
    )

    assert claimed is not None
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.started_at == original_started_at
    assert claimed.recovery_attempts == run.recovery_attempts + 1
    assert claimed.lease_id == "new-lease"
    assert claimed.lease_expires_at == expires_at
    assert claimed.output is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.lease_id == "new-lease"
    assert restored.lease_expires_at == expires_at


def test_fail_if_owner_rejects_expired_lease(repository) -> None:
    run = make_run(
        run_id="run-fail-expired",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "lease-a"
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 1, tzinfo=UTC)


@pytest.mark.skipif(
    __import__("os").getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)
def test_list_expired_running_runs_filters_before_limit(repository) -> None:
    stale_before = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

    repository.create(
        make_run(
            run_id="postgres-unexpired-1",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(
                2026,
                9,
                19,
                12,
                5,
                tzinfo=UTC,
            ),
        )
    )
    repository.create(
        make_run(
            run_id="postgres-unexpired-2",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(
                2026,
                9,
                19,
                12,
                6,
                tzinfo=UTC,
            ),
        )
    )
    repository.create(
        make_run(
            run_id="postgres-cancelled-stale",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(
                2026,
                9,
                19,
                11,
                57,
                tzinfo=UTC,
            ),
        ).model_copy(
            update={
                "cancellation_requested": True,
                "cancellation_requested_at": datetime(
                    2026,
                    9,
                    19,
                    11,
                    56,
                    tzinfo=UTC,
                ),
            }
        )
    )
    repository.create(
        make_run(
            run_id="postgres-stale-1",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(
                2026,
                9,
                19,
                11,
                58,
                tzinfo=UTC,
            ),
        )
    )
    repository.create(
        make_run(
            run_id="postgres-stale-2",
            status=AgentRunStatus.RUNNING,
            lease_expires_at=datetime(
                2026,
                9,
                19,
                11,
                59,
                tzinfo=UTC,
            ),
        )
    )

    runs = repository.list_expired_running_runs(
        stale_before=stale_before,
        limit=1,
    )

    assert [run.run_id for run in runs] == ["postgres-stale-1"]


@pytest.mark.skipif(
    __import__("os").getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)
def test_claim_expired_running_run_is_atomic_across_postgres_sessions() -> None:
    import os

    from sqlalchemy import create_engine, delete
    from sqlalchemy.orm import sessionmaker

    from app.control_plane.persistence.models import AgentRunRecord

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-stale-claim-race"
    stale_before = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)
    expired_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    started_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)
    lease_expires_at = datetime(2026, 9, 19, 12, 6, tzinfo=UTC)

    setup_session = session_factory()
    try:
        setup_repository = PostgreSQLAgentRunRepository(setup_session)
        setup_repository.create(
            make_run(
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                lease_id="expired-lease",
                lease_expires_at=expired_at,
            )
        )
    finally:
        setup_session.close()

    session_a = session_factory()
    session_b = session_factory()

    try:
        repository_a = PostgreSQLAgentRunRepository(session_a)
        repository_b = PostgreSQLAgentRunRepository(session_b)

        claimed_a = repository_a.claim_expired_running_run(
            run_id,
            stale_before=stale_before,
            started_at=started_at,
            lease_id="recovery-lease-a",
            lease_expires_at=lease_expires_at,
            max_recovery_attempts=3,
        )
        claimed_b = repository_b.claim_expired_running_run(
            run_id,
            stale_before=stale_before,
            started_at=started_at,
            lease_id="recovery-lease-b",
            lease_expires_at=lease_expires_at,
            max_recovery_attempts=3,
        )

        winners = [claimed for claimed in (claimed_a, claimed_b) if claimed is not None]

        assert len(winners) == 1
        assert winners[0].lease_id in {"recovery-lease-a", "recovery-lease-b"}

        with session_factory() as verification_session:
            verification_repository = PostgreSQLAgentRunRepository(verification_session)
            restored = verification_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.RUNNING
        assert restored.lease_id == winners[0].lease_id
        assert restored.lease_expires_at == lease_expires_at
    finally:
        session_a.close()
        session_b.close()

        cleanup_session = session_factory()
        try:
            cleanup_session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.run_id == run_id,
                )
            )
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            engine.dispose()


@pytest.mark.skipif(
    __import__("os").getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)
def test_stale_worker_cannot_complete_after_postgres_reclaim() -> None:
    import os

    from sqlalchemy import create_engine, delete
    from sqlalchemy.orm import sessionmaker

    from app.control_plane.persistence.models import AgentRunRecord

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-zombie-worker-after-reclaim"
    stale_before = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)
    expired_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    reclaimed_started_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)
    reclaimed_expires_at = datetime(2026, 9, 19, 12, 6, tzinfo=UTC)
    stale_worker_completed_at = datetime(2026, 9, 19, 12, 3, tzinfo=UTC)

    setup_session = session_factory()
    try:
        setup_repository = PostgreSQLAgentRunRepository(setup_session)
        setup_repository.create(
            make_run(
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                lease_id="lease-a",
                lease_expires_at=expired_at,
            )
        )
    finally:
        setup_session.close()

    stale_worker_session = session_factory()
    recovery_worker_session = session_factory()

    try:
        stale_worker_repository = PostgreSQLAgentRunRepository(stale_worker_session)
        recovery_worker_repository = PostgreSQLAgentRunRepository(recovery_worker_session)

        reclaimed = recovery_worker_repository.claim_expired_running_run(
            run_id,
            stale_before=stale_before,
            started_at=reclaimed_started_at,
            lease_id="lease-b",
            lease_expires_at=reclaimed_expires_at,
            max_recovery_attempts=3,
        )

        assert reclaimed is not None
        assert reclaimed.status is AgentRunStatus.RUNNING
        assert reclaimed.lease_id == "lease-b"
        assert reclaimed.lease_expires_at == reclaimed_expires_at

        stale_completion = stale_worker_repository.complete_if_owner(
            run_id,
            lease_id="lease-a",
            completed_at=stale_worker_completed_at,
            output={"answer": "zombie-worker-must-not-win"},
        )

        assert stale_completion is None

        with session_factory() as verification_session:
            verification_repository = PostgreSQLAgentRunRepository(verification_session)
            restored = verification_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.RUNNING
        assert restored.lease_id == "lease-b"
        assert restored.lease_expires_at == reclaimed_expires_at
        assert restored.completed_at is None
        assert restored.output is None
    finally:
        stale_worker_session.close()
        recovery_worker_session.close()

        cleanup_session = session_factory()
        try:
            cleanup_session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.run_id == run_id,
                )
            )
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            engine.dispose()


def test_complete_if_owner_persists_completed_run_and_clears_lease(repository) -> None:
    run = make_run(
        run_id="run-complete",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.COMPLETED
    assert restored.completed_at == completed_at
    assert restored.output == {"answer": "done"}
    assert restored.lease_id is None
    assert restored.lease_expires_at is None


def test_complete_if_owner_rejects_wrong_lease(repository) -> None:
    run = make_run(
        run_id="run-complete-wrong",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
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
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 5, tzinfo=UTC)


def test_complete_if_owner_rejects_expired_lease(repository) -> None:
    run = make_run(
        run_id="run-complete-expired",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "lease-a"
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 1, tzinfo=UTC)


def test_complete_if_owner_rejects_exact_expiry(repository) -> None:
    run = make_run(
        run_id="run-complete-exact-expiry",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "lease-a"
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 2, tzinfo=UTC)


def test_fail_if_owner_persists_failed_run_and_clears_lease(repository) -> None:
    run = make_run(
        run_id="run-fail",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.FAILED
    assert restored.completed_at == completed_at
    assert restored.error_type == "RuntimeError"
    assert restored.error_message == "agent failed"
    assert restored.lease_id is None
    assert restored.lease_expires_at is None


def test_fail_if_owner_rejects_wrong_lease(repository) -> None:
    run = make_run(
        run_id="run-fail-wrong",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
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
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 5, tzinfo=UTC)


@pytest.mark.skipif(
    __import__("os").getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)
def test_complete_if_owner_is_atomic_across_postgres_sessions() -> None:
    import os

    from sqlalchemy import create_engine, delete
    from sqlalchemy.orm import sessionmaker

    from app.control_plane.persistence.models import AgentRunRecord

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-terminal-ownership-race"
    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)
    lease_expires_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)

    setup_session = session_factory()
    try:
        setup_repository = PostgreSQLAgentRunRepository(setup_session)
        setup_repository.create(
            make_run(
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                lease_id="lease-a",
                lease_expires_at=lease_expires_at,
            )
        )
    finally:
        setup_session.close()

    session_a = session_factory()
    session_b = session_factory()

    try:
        repository_a = PostgreSQLAgentRunRepository(session_a)
        repository_b = PostgreSQLAgentRunRepository(session_b)

        completed_a = repository_a.complete_if_owner(
            run_id,
            lease_id="lease-a",
            completed_at=completed_at,
            output={"owner": "a"},
        )
        completed_b = repository_b.complete_if_owner(
            run_id,
            lease_id="lease-b",
            completed_at=completed_at,
            output={"owner": "b"},
        )

        assert completed_a is not None
        assert completed_a.status is AgentRunStatus.COMPLETED
        assert completed_a.lease_id is None
        assert completed_a.lease_expires_at is None

        assert completed_b is None

        with session_factory() as verification_session:
            verification_repository = PostgreSQLAgentRunRepository(verification_session)
            restored = verification_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.COMPLETED
        assert restored.output == {"owner": "a"}
        assert restored.lease_id is None
        assert restored.lease_expires_at is None
    finally:
        session_a.close()
        session_b.close()

        cleanup_session = session_factory()
        try:
            cleanup_session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.run_id == run_id,
                )
            )
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            engine.dispose()


def test_cancel_if_owner_persists_cancelled_run_and_clears_lease(repository) -> None:
    run = make_run(
        run_id="run-cancel",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
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


@pytest.mark.skipif(
    __import__("os").getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)
def test_cancel_if_owner_is_atomic_across_postgres_sessions() -> None:
    import os

    from sqlalchemy import create_engine, delete
    from sqlalchemy.orm import sessionmaker

    from app.control_plane.persistence.models import AgentRunRecord

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-cancel-ownership-race"
    cancelled_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)
    lease_expires_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)

    setup_session = session_factory()
    try:
        setup_repository = PostgreSQLAgentRunRepository(setup_session)
        setup_repository.create(
            make_run(
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                lease_id="lease-a",
                lease_expires_at=lease_expires_at,
            )
        )
    finally:
        setup_session.close()

    session_a = session_factory()
    session_b = session_factory()

    try:
        repository_a = PostgreSQLAgentRunRepository(session_a)
        repository_b = PostgreSQLAgentRunRepository(session_b)

        cancelled_a = repository_a.cancel_if_owner(
            run_id,
            lease_id="lease-a",
            completed_at=cancelled_at,
        )
        cancelled_b = repository_b.cancel_if_owner(
            run_id,
            lease_id="lease-b",
            completed_at=cancelled_at,
        )

        assert cancelled_a is not None
        assert cancelled_a.status is AgentRunStatus.CANCELLED
        assert cancelled_a.lease_id is None
        assert cancelled_a.lease_expires_at is None

        assert cancelled_b is None

        with session_factory() as verification_session:
            verification_repository = PostgreSQLAgentRunRepository(
                verification_session,
            )
            restored = verification_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.CANCELLED
        assert restored.completed_at == cancelled_at
        assert restored.lease_id is None
        assert restored.lease_expires_at is None
    finally:
        session_a.close()
        session_b.close()

        cleanup_session = session_factory()
        try:
            cleanup_session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.run_id == run_id,
                )
            )
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            engine.dispose()


def test_cancel_if_owner_rejects_wrong_lease(repository) -> None:
    run = make_run(
        run_id="run-cancel-wrong",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 5, tzinfo=UTC),
    )
    repository.create(run)

    result = repository.cancel_if_owner(
        run.run_id,
        lease_id="lease-b",
        completed_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert result is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "lease-a"
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 5, tzinfo=UTC)


def test_cancel_if_owner_rejects_expired_lease(repository) -> None:
    run = make_run(
        run_id="run-cancel-expired",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-a",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
    )
    repository.create(run)

    completed_at = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)

    result = repository.cancel_if_owner(
        run.run_id,
        lease_id="lease-a",
        completed_at=completed_at,
    )

    assert result is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.lease_id == "lease-a"
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 1, tzinfo=UTC)


def test_request_cancellation_persists_intent(repository) -> None:
    requested_at = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)

    run = make_run(
        run_id="run-cancel-request",
        status=AgentRunStatus.RUNNING,
    )
    repository.create(run)

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


def test_request_cancellation_is_idempotent_and_preserves_first_timestamp(
    repository,
) -> None:
    first_requested_at = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
    second_requested_at = datetime(2026, 9, 20, 12, 5, tzinfo=UTC)

    run = make_run(
        run_id="run-cancel-idempotent",
        status=AgentRunStatus.RUNNING,
    )
    repository.create(run)

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

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.cancellation_requested is True
    assert restored.cancellation_requested_at == first_requested_at


def test_request_cancellation_rejects_non_running_run(repository) -> None:
    requested_at = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)

    for status in (
        AgentRunStatus.PENDING,
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
        AgentRunStatus.CANCELLED,
        AgentRunStatus.REJECTED,
    ):
        run = make_run(
            run_id=f"cancel-{status.value}",
            status=status,
        )
        repository.create(run)

        result = repository.request_cancellation(
            run.run_id,
            requested_at=requested_at,
        )

        assert result is None

        restored = repository.get(run.run_id)
        assert restored is not None
        assert restored.cancellation_requested is False
        assert restored.cancellation_requested_at is None


def test_request_cancellation_rejects_missing_run(repository) -> None:
    result = repository.request_cancellation(
        "does-not-exist",
        requested_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
    )

    assert result is None


@pytest.mark.skipif(
    __import__("os").getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)
def test_concurrent_claim_expired_running_run_respects_max_recovery_attempts() -> None:
    import os

    from sqlalchemy import create_engine, delete
    from sqlalchemy.orm import sessionmaker

    from app.control_plane.persistence.models import AgentRunRecord

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-recovery-attempt-limit-race"
    max_recovery_attempts = 3
    stale_before = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)
    expired_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    started_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)
    lease_expires_at = datetime(2026, 9, 19, 12, 6, tzinfo=UTC)

    setup_session = session_factory()
    try:
        setup_repository = PostgreSQLAgentRunRepository(setup_session)
        setup_repository.create(
            make_run(
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                recovery_attempts=2,
                lease_id="expired-lease",
                lease_expires_at=expired_at,
            )
        )
    finally:
        setup_session.close()

    barrier = Barrier(5)

    def attempt_claim(worker_number: int):
        session = session_factory()

        try:
            repository = PostgreSQLAgentRunRepository(session)

            barrier.wait(timeout=10)

            return repository.claim_expired_running_run(
                run_id,
                stale_before=stale_before,
                started_at=started_at,
                lease_id=f"limit-race-lease-{worker_number}",
                lease_expires_at=lease_expires_at,
                max_recovery_attempts=max_recovery_attempts,
            )
        finally:
            session.close()

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(attempt_claim, range(5)))

        winners = [result for result in results if result is not None]

        assert len(winners) == 1
        assert winners[0].recovery_attempts == max_recovery_attempts

        with session_factory() as verification_session:
            verification_repository = PostgreSQLAgentRunRepository(
                verification_session,
            )
            restored = verification_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.RUNNING
        assert restored.recovery_attempts == max_recovery_attempts
        assert restored.recovery_attempts <= max_recovery_attempts
        assert restored.lease_id == winners[0].lease_id
        assert restored.lease_expires_at == lease_expires_at
    finally:
        cleanup_session = session_factory()
        try:
            cleanup_session.execute(delete(AgentRunRecord).where(AgentRunRecord.run_id == run_id))
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            engine.dispose()


@pytest.mark.skipif(
    __import__("os").getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)
def test_concurrent_claim_expired_running_run_is_atomic() -> None:
    import os

    from sqlalchemy import create_engine, delete
    from sqlalchemy.orm import sessionmaker

    from app.control_plane.persistence.models import AgentRunRecord

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-concurrent-stale-claim-race"
    stale_before = datetime(2026, 9, 19, 12, 2, tzinfo=UTC)
    expired_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)
    started_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)
    lease_expires_at = datetime(2026, 9, 19, 12, 6, tzinfo=UTC)

    setup_session = session_factory()
    try:
        setup_repository = PostgreSQLAgentRunRepository(setup_session)
        setup_repository.create(
            make_run(
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                lease_id="expired-lease",
                lease_expires_at=expired_at,
            )
        )
    finally:
        setup_session.close()

    barrier = Barrier(5)

    def attempt_claim(worker_number: int):
        session = session_factory()

        try:
            repository = PostgreSQLAgentRunRepository(session)

            barrier.wait(timeout=10)

            return repository.claim_expired_running_run(
                run_id,
                stale_before=stale_before,
                started_at=started_at,
                lease_id=f"concurrent-recovery-lease-{worker_number}",
                lease_expires_at=lease_expires_at,
                max_recovery_attempts=3,
            )
        finally:
            session.close()

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(attempt_claim, range(5)))

        winners = [result for result in results if result is not None]

        assert len(winners) == 1
        assert winners[0].lease_id.startswith("concurrent-recovery-lease-")

        with session_factory() as verification_session:
            verification_repository = PostgreSQLAgentRunRepository(verification_session)
            restored = verification_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.RUNNING
        assert restored.lease_id == winners[0].lease_id
        assert restored.lease_expires_at == lease_expires_at
    finally:
        cleanup_session = session_factory()
        try:
            cleanup_session.execute(delete(AgentRunRecord).where(AgentRunRecord.run_id == run_id))
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            engine.dispose()


@pytest.mark.skipif(
    __import__("os").getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)
def test_concurrent_claim_for_recovery_is_atomic() -> None:
    import os

    from sqlalchemy import create_engine, delete
    from sqlalchemy.orm import sessionmaker

    from app.control_plane.persistence.models import AgentRunRecord

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "pg-concurrent-recovery-race"
    started_at = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)
    lease_expires_at = datetime(2026, 9, 19, 12, 6, tzinfo=UTC)

    setup_session = session_factory()
    try:
        setup_repository = PostgreSQLAgentRunRepository(setup_session)
        setup_repository.create(
            make_run(
                run_id=run_id,
                status=AgentRunStatus.FAILED,
                started_at=datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
                completed_at=datetime(2026, 9, 17, 10, 5, tzinfo=UTC),
                error_type="RuntimeError",
                error_message="previous attempt failed",
                output={"partial": "output"},
                metadata={"attempt": 1},
            )
        )
    finally:
        setup_session.close()

    barrier = Barrier(5)

    def attempt_recovery(worker_number: int):
        session = session_factory()

        try:
            repository = PostgreSQLAgentRunRepository(session)

            barrier.wait(timeout=10)

            return repository.claim_for_recovery(
                run_id,
                started_at=started_at,
                lease_id=f"concurrent-recovery-lease-{worker_number}",
                lease_expires_at=lease_expires_at,
                max_recovery_attempts=3,
            )
        finally:
            session.close()

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(attempt_recovery, range(5)))

        winners = [result for result in results if result is not None]

        assert len(winners) == 1
        assert winners[0].status is AgentRunStatus.RUNNING
        assert winners[0].lease_id.startswith("concurrent-recovery-lease-")

        with session_factory() as verification_session:
            verification_repository = PostgreSQLAgentRunRepository(verification_session)
            restored = verification_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.RUNNING
        assert restored.lease_id == winners[0].lease_id
        assert restored.lease_expires_at == lease_expires_at
        assert restored.error_type is None
        assert restored.error_message is None
        assert restored.output is None
    finally:
        cleanup_session = session_factory()
        try:
            cleanup_session.execute(delete(AgentRunRecord).where(AgentRunRecord.run_id == run_id))
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            engine.dispose()


def test_create_and_get_round_trip_preserves_idempotency_key(repository) -> None:
    run = make_run(
        run_id="idempotency-round-trip",
        idempotency_key="request-key-1",
    )

    repository.create(run)

    restored = repository.get(run.run_id)

    assert restored is not None
    assert restored.idempotency_key == "request-key-1"
    assert restored == run


def test_get_by_idempotency_key_returns_matching_user_run(repository) -> None:
    run = make_run(
        run_id="idempotency-lookup",
        user_id="user-42",
        idempotency_key="request-key-42",
    )
    repository.create(run)

    restored = repository.get_by_idempotency_key(
        "user-42",
        "request-key-42",
    )

    assert restored == run


def test_get_by_idempotency_key_is_scoped_to_user(repository) -> None:
    run = make_run(
        run_id="idempotency-user-scope",
        user_id="user-1",
        idempotency_key="shared-key",
    )
    repository.create(run)

    assert repository.get_by_idempotency_key("user-2", "shared-key") is None


def test_get_by_idempotency_key_returns_none_for_missing_key(repository) -> None:
    assert (
        repository.get_by_idempotency_key(
            "user-1",
            "does-not-exist",
        )
        is None
    )


def test_update_round_trip_preserves_idempotency_key(repository) -> None:
    repository.create(
        make_run(
            run_id="idempotency-update",
            idempotency_key="old-key",
        )
    )

    updated = make_run(
        run_id="idempotency-update",
        idempotency_key="new-key",
        status=AgentRunStatus.COMPLETED,
        output={"answer": "updated"},
    )

    result = repository.update(updated)
    restored = repository.get(updated.run_id)

    assert result == updated
    assert restored is not None
    assert restored.idempotency_key == "new-key"


def test_duplicate_user_and_idempotency_key_is_rejected(repository) -> None:
    repository.create(
        make_run(
            run_id="idempotency-duplicate-1",
            user_id="user-1",
            idempotency_key="duplicate-key",
        )
    )

    with pytest.raises(DuplicateAgentRunError):
        repository.create(
            make_run(
                run_id="idempotency-duplicate-2",
                user_id="user-1",
                idempotency_key="duplicate-key",
            )
        )


def test_same_idempotency_key_is_allowed_for_different_users(repository) -> None:
    first = make_run(
        run_id="idempotency-user-1",
        user_id="user-1",
        idempotency_key="same-key",
    )
    second = make_run(
        run_id="idempotency-user-2",
        user_id="user-2",
        idempotency_key="same-key",
    )

    repository.create(first)
    repository.create(second)

    assert repository.get_by_idempotency_key("user-1", "same-key") == first
    assert repository.get_by_idempotency_key("user-2", "same-key") == second


def test_multiple_null_idempotency_keys_are_allowed(repository) -> None:
    first = make_run(
        run_id="non-idempotent-1",
        idempotency_key=None,
    )
    second = make_run(
        run_id="non-idempotent-2",
        idempotency_key=None,
    )

    repository.create(first)
    repository.create(second)

    assert repository.get(first.run_id) == first
    assert repository.get(second.run_id) == second
