"""PostgreSQL integration tests for durable approval requests."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from app.control_plane.agent_runs.models import AgentRun
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.approvals.models import ApprovalRequest, ApprovalStatus
from app.control_plane.approvals.postgres_repository import (
    PostgreSQLApprovalRequestRepository,
)
from app.control_plane.persistence.database import SessionLocal

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


def make_run(
    *,
    run_id: str,
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name="approval-integration-agent",
        session_id=f"session-{run_id}",
        user_id="approval-integration-user",
        principal="api_key:approval-integration",
        tenant_id="tenant-approval",
    )


def make_approval(
    *,
    approval_id: str = "approval-1",
    run_id: str = "approval-run-1",
    step_id: str = "step-1",
    call_id: str = "call-1",
    idempotency_key: str = "idem-1",
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        run_id=run_id,
        step_id=step_id,
        call_id=call_id,
        tool_name="send_notification",
        idempotency_key=idempotency_key,
        policy_name="side-effect-requires-approval",
        policy_version="1.2.0",
        risk_tier="high",
        requested_action="Send an external notification",
        policy_metadata={
            "tenant_id": "tenant-approval",
            "requires_human_approval": True,
            "source": "integration-test",
        },
        created_at=datetime(
            2026,
            9,
            26,
            9,
            0,
            tzinfo=UTC,
        ),
        updated_at=datetime(
            2026,
            9,
            26,
            9,
            0,
            tzinfo=UTC,
        ),
    )


def test_postgres_approval_create_get_round_trip() -> None:
    run_id = "approval-pg-round-trip"
    approval = make_approval(
        approval_id="approval-pg-round-trip",
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            repository = PostgreSQLApprovalRequestRepository(session)
            created = repository.create(approval)
            restored = repository.get(approval.approval_id)

            assert created == approval
            assert restored is not None
            assert restored == approval
            assert restored.status is ApprovalStatus.PENDING
            assert restored.policy_metadata == approval.policy_metadata
            assert restored.created_at is not None
            assert restored.updated_at is not None
            assert restored.resolved_at is None
    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import (
                AgentRunRecord,
            )

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_resolution_persists() -> None:
    run_id = "approval-pg-resolution"
    approval = make_approval(
        approval_id="approval-pg-resolution",
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            repository = PostgreSQLApprovalRequestRepository(session)
            repository.create(approval)

            resolved = repository.update_status(
                approval.approval_id,
                ApprovalStatus.APPROVED,
                resolved_by="human-reviewer-1",
                resolution_reason="Approved for controlled execution.",
            )

            assert resolved.status is ApprovalStatus.APPROVED
            assert resolved.resolved_by == "human-reviewer-1"
            assert resolved.resolution_reason == ("Approved for controlled execution.")
            assert resolved.resolved_at is not None

        with SessionLocal() as verification_session:
            repository = PostgreSQLApprovalRequestRepository(verification_session)
            restored = repository.get(approval.approval_id)

            assert restored is not None
            assert restored.status is ApprovalStatus.APPROVED
            assert restored.resolved_by == "human-reviewer-1"
            assert restored.resolution_reason == ("Approved for controlled execution.")
            assert restored.resolved_at is not None
    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import (
                AgentRunRecord,
            )

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_cannot_be_resolved_twice() -> None:
    run_id = "approval-pg-double-resolution"
    approval = make_approval(
        approval_id="approval-pg-double-resolution",
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            repository = PostgreSQLApprovalRequestRepository(session)
            repository.create(approval)

            repository.update_status(
                approval.approval_id,
                ApprovalStatus.REJECTED,
                resolved_by="human-reviewer-2",
                resolution_reason="Rejected by reviewer.",
            )

            with pytest.raises(
                ValueError,
                match="approval request can only be resolved from pending status",
            ):
                repository.update_status(
                    approval.approval_id,
                    ApprovalStatus.APPROVED,
                    resolved_by="human-reviewer-3",
                )
    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import (
                AgentRunRecord,
            )

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_list_by_run_is_deterministic() -> None:
    run_id = "approval-pg-list"
    first = make_approval(
        approval_id="approval-pg-list-a",
        run_id=run_id,
        step_id="step-a",
        call_id="call-a",
        idempotency_key="idem-a",
    )
    second = make_approval(
        approval_id="approval-pg-list-b",
        run_id=run_id,
        step_id="step-b",
        call_id="call-b",
        idempotency_key="idem-b",
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            repository = PostgreSQLApprovalRequestRepository(session)
            repository.create(
                first.model_copy(
                    update={
                        "created_at": datetime(
                            2026,
                            9,
                            26,
                            10,
                            0,
                            tzinfo=UTC,
                        ),
                        "updated_at": datetime(
                            2026,
                            9,
                            26,
                            10,
                            0,
                            tzinfo=UTC,
                        ),
                    }
                )
            )
            repository.create(
                second.model_copy(
                    update={
                        "created_at": datetime(
                            2026,
                            9,
                            26,
                            10,
                            1,
                            tzinfo=UTC,
                        ),
                        "updated_at": datetime(
                            2026,
                            9,
                            26,
                            10,
                            1,
                            tzinfo=UTC,
                        ),
                    }
                )
            )

            listed = repository.list_by_run(run_id)

            assert [item.approval_id for item in listed] == [
                "approval-pg-list-a",
                "approval-pg-list-b",
            ]
    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import (
                AgentRunRecord,
            )

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_rejects_duplicate_approval_id() -> None:
    run_id = "approval-pg-duplicate-id"
    approval = make_approval(
        approval_id="approval-pg-duplicate-id",
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            repository = PostgreSQLApprovalRequestRepository(session)
            repository.create(approval)

            with pytest.raises(
                ValueError,
                match="approval request already exists: approval-pg-duplicate-id",
            ):
                repository.create(approval)
    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import (
                AgentRunRecord,
            )

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_enforces_run_step_call_uniqueness() -> None:
    run_id = "approval-pg-unique-call"
    first = make_approval(
        approval_id="approval-pg-unique-call-a",
        run_id=run_id,
        step_id="step-1",
        call_id="call-1",
    )
    duplicate_identity = make_approval(
        approval_id="approval-pg-unique-call-b",
        run_id=run_id,
        step_id="step-1",
        call_id="call-1",
        idempotency_key="idem-2",
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            repository = PostgreSQLApprovalRequestRepository(session)
            repository.create(first)

            with pytest.raises(IntegrityError):
                repository.create(duplicate_identity)

            session.rollback()

            assert repository.get(first.approval_id) is not None
            assert repository.get(duplicate_identity.approval_id) is None
    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import (
                AgentRunRecord,
            )

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_create_can_be_rolled_back() -> None:
    run_id = "approval-pg-rollback"
    approval = make_approval(
        approval_id="approval-pg-rollback",
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            repository = PostgreSQLApprovalRequestRepository(session)
            repository.create(approval, commit=False)

            assert repository.get(approval.approval_id) == approval

            session.rollback()

            assert repository.get(approval.approval_id) is None
    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import (
                AgentRunRecord,
            )

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()
