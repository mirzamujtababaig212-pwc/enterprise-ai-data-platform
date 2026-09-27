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
from app.control_plane.approvals.models import ApprovalOverride, ApprovalRequest, ApprovalStatus
from app.control_plane.approvals.postgres_repository import (
    PostgreSQLApprovalOverrideRepository,
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
    principal: str = "api_key:approval-integration",
    tenant_id: str = "tenant-approval",
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name="approval-integration-agent",
        session_id=f"session-{run_id}",
        user_id="approval-integration-user",
        principal=principal,
        tenant_id=tenant_id,
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


def make_override(
    *,
    override_id: str = "override-1",
    approval_id: str = "approval-1",
    run_id: str = "approval-run-1",
    created_at: datetime | None = None,
) -> ApprovalOverride:
    return ApprovalOverride(
        override_id=override_id,
        approval_id=approval_id,
        run_id=run_id,
        actor="operator-1",
        reason="Emergency operational bypass.",
        created_at=created_at
        or datetime(
            2026,
            9,
            26,
            9,
            30,
            tzinfo=UTC,
        ),
    )


def test_postgres_approval_list_scopes_by_identity_status_and_limit() -> None:
    matching_run_id = "approval-pg-inbox-match"
    other_principal_run_id = "approval-pg-inbox-principal"
    other_tenant_run_id = "approval-pg-inbox-tenant"

    matching_pending = make_approval(
        approval_id="approval-pg-inbox-new",
        run_id=matching_run_id,
        step_id="step-new",
        call_id="call-new",
        idempotency_key="idem-new",
    ).model_copy(
        update={
            "created_at": datetime(2026, 9, 26, 11, 0, tzinfo=UTC),
            "updated_at": datetime(2026, 9, 26, 11, 0, tzinfo=UTC),
        }
    )

    matching_approved = make_approval(
        approval_id="approval-pg-inbox-approved",
        run_id=matching_run_id,
        step_id="step-approved",
        call_id="call-approved",
        idempotency_key="idem-approved",
    ).model_copy(
        update={
            "created_at": datetime(2026, 9, 26, 10, 0, tzinfo=UTC),
            "updated_at": datetime(2026, 9, 26, 10, 0, tzinfo=UTC),
            "status": ApprovalStatus.APPROVED,
            "resolved_by": "reviewer-1",
            "resolved_at": datetime(2026, 9, 26, 10, 30, tzinfo=UTC),
        }
    )

    other_principal = make_approval(
        approval_id="approval-pg-inbox-wrong-principal",
        run_id=other_principal_run_id,
        step_id="step-principal",
        call_id="call-principal",
        idempotency_key="idem-principal",
    ).model_copy(
        update={
            "created_at": datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
            "updated_at": datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
        }
    )

    other_tenant = make_approval(
        approval_id="approval-pg-inbox-wrong-tenant",
        run_id=other_tenant_run_id,
        step_id="step-tenant",
        call_id="call-tenant",
        idempotency_key="idem-tenant",
    ).model_copy(
        update={
            "created_at": datetime(2026, 9, 26, 13, 0, tzinfo=UTC),
            "updated_at": datetime(2026, 9, 26, 13, 0, tzinfo=UTC),
        }
    )

    run_ids = {
        matching_run_id,
        other_principal_run_id,
        other_tenant_run_id,
    }

    try:
        with SessionLocal() as session:
            run_repository = PostgreSQLAgentRunRepository(session)
            repository = PostgreSQLApprovalRequestRepository(session)

            run_repository.create(make_run(run_id=matching_run_id))
            run_repository.create(
                make_run(
                    run_id=other_principal_run_id,
                    principal="api_key:other-principal",
                )
            )
            run_repository.create(
                make_run(
                    run_id=other_tenant_run_id,
                    tenant_id="tenant-other",
                )
            )

            repository.create(matching_pending)
            repository.create(matching_approved)
            repository.create(other_principal)
            repository.create(other_tenant)

            pending = repository.list(
                tenant_id="tenant-approval",
                principal="api_key:approval-integration",
                status=ApprovalStatus.PENDING,
            )

            assert [item.approval_id for item in pending] == [
                "approval-pg-inbox-new",
            ]

            scoped = repository.list(
                tenant_id="tenant-approval",
                principal="api_key:approval-integration",
                limit=10,
            )

            assert [item.approval_id for item in scoped] == [
                "approval-pg-inbox-new",
                "approval-pg-inbox-approved",
            ]

            limited = repository.list(
                tenant_id="tenant-approval",
                principal="api_key:approval-integration",
                limit=1,
            )

            assert [item.approval_id for item in limited] == [
                "approval-pg-inbox-new",
            ]

            tenant_only = repository.list(
                tenant_id="tenant-approval",
                limit=10,
            )

            assert [item.approval_id for item in tenant_only] == [
                "approval-pg-inbox-wrong-principal",
                "approval-pg-inbox-new",
                "approval-pg-inbox-approved",
            ]

            principal_only = repository.list(
                principal="api_key:approval-integration",
                limit=10,
            )

            assert [item.approval_id for item in principal_only] == [
                "approval-pg-inbox-wrong-tenant",
                "approval-pg-inbox-new",
                "approval-pg-inbox-approved",
            ]

    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import AgentRunRecord

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id.in_(run_ids)).delete(
                synchronize_session=False
            )
            session.commit()


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


def test_postgres_approval_override_create_get_round_trip() -> None:
    run_id = "approval-pg-override-round-trip"
    approval = make_approval(
        approval_id="approval-pg-override-round-trip",
        run_id=run_id,
    )
    override = make_override(
        override_id="override-pg-round-trip",
        approval_id=approval.approval_id,
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            approval_repository = PostgreSQLApprovalRequestRepository(session)
            approval_repository.create(approval)

            repository = PostgreSQLApprovalOverrideRepository(session)
            created = repository.create(override)
            restored = repository.get(override.override_id)
            by_approval = repository.get_by_approval(approval.approval_id)

            assert created == override
            assert restored == override
            assert by_approval == override

    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import AgentRunRecord

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_override_list_by_run_is_deterministic() -> None:
    run_id = "approval-pg-override-list"

    approval_a = make_approval(
        approval_id="approval-pg-override-list-a",
        run_id=run_id,
        step_id="step-a",
        call_id="call-a",
        idempotency_key="idem-a",
    )
    approval_b = make_approval(
        approval_id="approval-pg-override-list-b",
        run_id=run_id,
        step_id="step-b",
        call_id="call-b",
        idempotency_key="idem-b",
    )

    first = make_override(
        override_id="override-pg-list-a",
        approval_id=approval_a.approval_id,
        run_id=run_id,
        created_at=datetime(2026, 9, 26, 10, 0, tzinfo=UTC),
    )
    second = make_override(
        override_id="override-pg-list-b",
        approval_id=approval_b.approval_id,
        run_id=run_id,
        created_at=datetime(2026, 9, 26, 10, 1, tzinfo=UTC),
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            approval_repository = PostgreSQLApprovalRequestRepository(session)
            approval_repository.create(approval_a)
            approval_repository.create(approval_b)

            repository = PostgreSQLApprovalOverrideRepository(session)
            repository.create(first)
            repository.create(second)

            listed = repository.list_by_run(run_id)

            assert [item.override_id for item in listed] == [
                "override-pg-list-a",
                "override-pg-list-b",
            ]

    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import AgentRunRecord

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_override_rejects_duplicate_override_id() -> None:
    run_id = "approval-pg-override-duplicate-id"
    approval = make_approval(
        approval_id="approval-pg-override-duplicate-id",
        run_id=run_id,
    )
    override = make_override(
        override_id="override-pg-duplicate-id",
        approval_id=approval.approval_id,
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            PostgreSQLApprovalRequestRepository(session).create(approval)

            repository = PostgreSQLApprovalOverrideRepository(session)
            repository.create(override)

            with pytest.raises(
                ValueError,
                match="approval override already exists",
            ):
                repository.create(override)

    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import AgentRunRecord

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_override_allows_only_one_override_per_approval() -> None:
    run_id = "approval-pg-override-unique-approval"
    approval = make_approval(
        approval_id="approval-pg-override-unique-approval",
        run_id=run_id,
    )
    first = make_override(
        override_id="override-pg-unique-approval-a",
        approval_id=approval.approval_id,
        run_id=run_id,
    )
    second = make_override(
        override_id="override-pg-unique-approval-b",
        approval_id=approval.approval_id,
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            PostgreSQLApprovalRequestRepository(session).create(approval)

            repository = PostgreSQLApprovalOverrideRepository(session)
            repository.create(first)

            with pytest.raises(
                ValueError,
                match="approval override already exists for approval",
            ):
                repository.create(second)

    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import AgentRunRecord

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_override_create_can_be_rolled_back() -> None:
    run_id = "approval-pg-override-rollback"
    approval = make_approval(
        approval_id="approval-pg-override-rollback",
        run_id=run_id,
    )
    override = make_override(
        override_id="override-pg-rollback",
        approval_id=approval.approval_id,
        run_id=run_id,
    )

    try:
        with SessionLocal() as session:
            PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))

            PostgreSQLApprovalRequestRepository(session).create(approval)

            repository = PostgreSQLApprovalOverrideRepository(session)
            repository.create(override, commit=False)

            assert repository.get(override.override_id) == override

            session.rollback()

            assert repository.get(override.override_id) is None

    finally:
        with SessionLocal() as session:
            from app.control_plane.persistence.models import AgentRunRecord

            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                synchronize_session=False
            )
            session.commit()


def test_postgres_approval_override_is_removed_with_agent_run() -> None:
    run_id = "approval-pg-override-cascade"
    approval = make_approval(
        approval_id="approval-pg-override-cascade",
        run_id=run_id,
    )
    override = make_override(
        override_id="override-pg-cascade",
        approval_id=approval.approval_id,
        run_id=run_id,
    )

    with SessionLocal() as session:
        PostgreSQLAgentRunRepository(session).create(make_run(run_id=run_id))
        PostgreSQLApprovalRequestRepository(session).create(approval)
        PostgreSQLApprovalOverrideRepository(session).create(override)

    with SessionLocal() as session:
        from app.control_plane.persistence.models import AgentRunRecord

        session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
            synchronize_session=False
        )
        session.commit()

    with SessionLocal() as session:
        assert PostgreSQLApprovalOverrideRepository(session).get(override.override_id) is None
