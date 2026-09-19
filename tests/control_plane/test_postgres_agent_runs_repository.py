from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.control_plane.agent_runs.exceptions import (
    AgentRunNotFoundError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
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
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        session_id=session_id,
        user_id=user_id,
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
    )

    result = repository.create(run)
    restored = repository.get(run.run_id)

    assert result == run
    assert restored is not None
    assert restored.run_id == run.run_id
    assert restored.agent_name == run.agent_name
    assert restored.session_id == run.session_id
    assert restored.user_id == run.user_id
    assert restored.status == AgentRunStatus.RUNNING
    assert restored.started_at == started_at
    assert restored.output == {"step": "started"}
    assert restored.metadata == {"source": "test", "attempt": 1}


def test_request_snapshot_round_trip_reconstructs_request(repository) -> None:
    request = AgentRequest(
        input="Find vehicle incidents for fleet-42",
        session_id="session-123",
        user_id="user-456",
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
    )

    snapshot = AgentRunRequestSnapshot.from_request(request)

    run = make_run(
        run_id="snapshot-round-trip",
        session_id=request.session_id,
        user_id=request.user_id,
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
    )

    assert reconstructed.input == request.input
    assert reconstructed.session_id == request.session_id
    assert reconstructed.user_id == request.user_id
    assert reconstructed.memory_namespace == request.memory_namespace
    assert reconstructed.governance_policy is not None
    assert reconstructed.governance_policy.required_metadata == (
        request.governance_policy.required_metadata
    )
    assert reconstructed.metadata == request.metadata


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
    )

    assert claimed is not None
    assert claimed.run_id == failed.run_id
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.started_at == recovery_started_at
    assert claimed.completed_at is None
    assert claimed.error_type is None
    assert claimed.error_message is None
    assert claimed.output is None
    assert claimed.metadata == failed.metadata

    restored = repository.get(failed.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.started_at == recovery_started_at


def test_claim_for_recovery_returns_none_for_non_failed_run(repository) -> None:
    running = make_run(status=AgentRunStatus.RUNNING)
    repository.create(running)

    claimed = repository.claim_for_recovery(
        running.run_id,
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
    )

    assert claimed is None

    restored = repository.get(running.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING


def test_claim_for_recovery_returns_none_for_missing_run(repository) -> None:
    claimed = repository.claim_for_recovery(
        "does-not-exist",
        started_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        lease_id="lease-1",
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
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
    )
    second_claim = repository.claim_for_recovery(
        failed.run_id,
        started_at=second_started_at,
        lease_id="lease-2",
        lease_expires_at=datetime(2026, 9, 19, 12, 2, tzinfo=UTC),
    )

    assert first_claim is not None
    assert first_claim.status is AgentRunStatus.RUNNING
    assert first_claim.started_at == first_started_at

    assert second_claim is None

    restored = repository.get(failed.run_id)
    assert restored is not None
    assert restored.status is AgentRunStatus.RUNNING
    assert restored.started_at == first_started_at


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
    )

    assert claimed is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.lease_id == "old-lease"


def test_expired_running_run_can_be_reclaimed(repository) -> None:
    run = make_run(
        status=AgentRunStatus.RUNNING,
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
    )

    assert claimed is not None
    assert claimed.status is AgentRunStatus.RUNNING
    assert claimed.started_at == started_at
    assert claimed.lease_id == "new-lease"
    assert claimed.lease_expires_at == expires_at
    assert claimed.output is None

    restored = repository.get(run.run_id)
    assert restored is not None
    assert restored.lease_id == "new-lease"
    assert restored.lease_expires_at == expires_at


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
        )
        claimed_b = repository_b.claim_expired_running_run(
            run_id,
            stale_before=stale_before,
            started_at=started_at,
            lease_id="recovery-lease-b",
            lease_expires_at=lease_expires_at,
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


def test_complete_if_owner_persists_completed_run_and_clears_lease(repository) -> None:
    run = make_run(
        run_id="run-complete",
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
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
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
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 1, tzinfo=UTC)


def test_fail_if_owner_persists_failed_run_and_clears_lease(repository) -> None:
    run = make_run(
        run_id="run-fail",
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
        lease_expires_at=datetime(2026, 9, 19, 12, 1, tzinfo=UTC),
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
    assert restored.lease_expires_at == datetime(2026, 9, 19, 12, 1, tzinfo=UTC)


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
    lease_expires_at = datetime(2026, 9, 19, 12, 1, tzinfo=UTC)

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
