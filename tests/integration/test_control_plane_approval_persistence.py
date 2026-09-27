"""PostgreSQL integration tests for durable approval requests."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_messages import user_message
from ai_platform.agents.models import AgentRequest, AgentResponse
from ai_platform.agents.observability import AgentExecutionEventType
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)
from app.control_plane.agent_run_events.postgres_observer import (
    PostgreSQLAgentRunEventObserver,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_steps.postgres_repository import (
    PostgreSQLAgentRunStepsRepository,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.agent_runs.request_snapshot import (
    AgentRunRequestSnapshot,
)
from app.control_plane.approvals.continuation_service import (
    AgentRunApprovalContinuationService,
)
from app.control_plane.approvals.models import ApprovalOverride, ApprovalRequest, ApprovalStatus
from app.control_plane.approvals.postgres_repository import (
    PostgreSQLApprovalOverrideRepository,
    PostgreSQLApprovalRequestRepository,
)
from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    AgentRunEventRecord,
    AgentRunRecord,
    AgentRunStepRecord,
    ApprovalRequestRecord,
)

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


class ApprovalContinuationRuntime:
    def __init__(self, session_factory, *, run_id: str, step_id: str) -> None:
        self._session_factory = session_factory
        self._run_id = run_id
        self._step_id = step_id
        self.calls: list[dict[str, object]] = []

    async def resume(
        self,
        agent_name,
        request,
        checkpoint,
        *,
        run_id=None,
        lease_id=None,
        execution_ownership_lost=None,
    ):
        self.calls.append(
            {
                "agent_name": agent_name,
                "request": request,
                "checkpoint": checkpoint,
                "run_id": run_id,
                "lease_id": lease_id,
            }
        )

        completed_at = datetime.now(UTC)
        with self._session_factory() as session:
            step_repository = PostgreSQLAgentRunStepsRepository(session)
            step = step_repository.transition(
                self._run_id,
                self._step_id,
                status=AgentRunStepStatus.COMPLETED,
                updated_at=completed_at,
                completed_at=completed_at,
                output={"status": "notification_sent"},
            )
            assert step is not None

        return AgentResponse(
            agent_name=agent_name,
            output="Notification sent after approval.",
            session_id=request.session_id,
        )


def test_postgres_approval_continuation_persists_full_approval_lifecycle() -> None:
    """Exercise approval decision, continuation, completion, and audit persistence."""

    session_factory = SessionLocal

    run_id = "pg-approval-continuation-e2e"
    approval_id = "pg-approval-continuation"
    step_id = "pg-approval-step"
    call_id = "pg-approval-call"
    principal = "api_key:approval-continuation-principal"
    tenant_id = "tenant-approval-continuation"

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        step_repository = PostgreSQLAgentRunStepsRepository(session)
        checkpoint_repository = PostgreSQLAgentCheckpointsRepository(session)
        approval_repository = PostgreSQLApprovalRequestRepository(session)

        request = AgentRequest(
            input="Send the approved fleet notification.",
            session_id="session-approval-continuation",
            user_id="approval-continuation-user",
            principal=principal,
            tenant_id=tenant_id,
            memory_namespace="fleet-memory",
            metadata={
                "source": "approval-continuation-integration",
            },
        )

        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="approval-continuation-agent",
                session_id=request.session_id,
                user_id=request.user_id,
                principal=request.principal,
                tenant_id=request.tenant_id,
                status=AgentRunStatus.WAITING_FOR_APPROVAL,
                started_at=datetime.now(UTC),
                request_snapshot=AgentRunRequestSnapshot.from_request(request),
            )
        )

        step_repository.create(
            AgentRunStep(
                run_id=run_id,
                step_id=step_id,
                step_index=0,
                step_type="tool_call",
                status=AgentRunStepStatus.RUNNING,
                attempt=1,
                tool_name="send_notification",
                call_id=call_id,
            )
        )

        checkpoint = AgentExecutionCheckpoint(
            schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
            run_id=run_id,
            agent_name="approval-continuation-agent",
            session_id=request.session_id,
            user_id=request.user_id,
            messages=(user_message("Send the approved fleet notification."),),
            tool_round=1,
            position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
            metadata={
                "source": "approval-continuation-integration",
            },
        )
        checkpoint_repository.save(checkpoint)

        approval_repository.create(
            ApprovalRequest(
                approval_id=approval_id,
                run_id=run_id,
                step_id=step_id,
                call_id=call_id,
                tool_name="send_notification",
                idempotency_key="pg-approval-continuation-idem",
                status=ApprovalStatus.PENDING,
                policy_name="side-effect-requires-approval",
                policy_version="1.2.0",
                risk_tier="high",
                requested_action="Send the approved fleet notification",
                policy_metadata={
                    "requires_human_approval": True,
                    "source": "approval-continuation-integration",
                },
            )
        )

        runtime = ApprovalContinuationRuntime(
            session_factory,
            run_id=run_id,
            step_id=step_id,
        )
        observer = PostgreSQLAgentRunEventObserver(session_factory)

        continuation_service = AgentRunApprovalContinuationService(
            runtime=runtime,
            approval_repository=approval_repository,
            agent_run_repository=run_repository,
            agent_run_steps_repository=step_repository,
            checkpoints_repository=checkpoint_repository,
            lease_seconds=60,
            observer=observer,
        )

        import asyncio

        response = asyncio.run(
            continuation_service.continue_approval(
                approval_id,
                status=ApprovalStatus.APPROVED,
                resolved_by="approver-integration",
                resolution_reason="Approved for execution.",
            )
        )

        assert response.output == "Notification sent after approval."

        assert len(runtime.calls) == 1
        runtime_call = runtime.calls[0]
        assert runtime_call["agent_name"] == "approval-continuation-agent"
        assert runtime_call["run_id"] == run_id
        resumed_checkpoint = runtime_call["checkpoint"]
        assert resumed_checkpoint.run_id == checkpoint.run_id
        assert resumed_checkpoint.agent_name == checkpoint.agent_name
        assert resumed_checkpoint.session_id == checkpoint.session_id
        assert resumed_checkpoint.user_id == checkpoint.user_id
        assert resumed_checkpoint.messages == checkpoint.messages
        assert resumed_checkpoint.tool_round == checkpoint.tool_round
        assert resumed_checkpoint.position == checkpoint.position
        assert resumed_checkpoint.metadata == checkpoint.metadata
        assert resumed_checkpoint.execution_budget_state.llm_calls == (
            checkpoint.execution_budget_state.llm_calls
        )
        assert resumed_checkpoint.execution_budget_state.tool_calls == (
            checkpoint.execution_budget_state.tool_calls
        )
        assert resumed_checkpoint.execution_budget_state.tool_rounds == (
            checkpoint.execution_budget_state.tool_rounds
        )
        assert resumed_checkpoint.execution_budget_state.total_tokens == (
            checkpoint.execution_budget_state.total_tokens
        )
        assert resumed_checkpoint.execution_budget_state.elapsed_seconds >= 0

        resumed_request = runtime_call["request"]
        assert resumed_request.input == request.input
        assert resumed_request.session_id == request.session_id
        assert resumed_request.user_id == request.user_id
        assert resumed_request.principal == principal
        assert resumed_request.tenant_id == tenant_id
        assert resumed_request.memory_namespace == "fleet-memory"
        assert resumed_request.metadata == {
            "source": "approval-continuation-integration",
        }

        session.rollback()

        restored_approval = approval_repository.get(approval_id)
        assert restored_approval is not None
        assert restored_approval.status is ApprovalStatus.APPROVED
        assert restored_approval.resolved_by == "approver-integration"
        assert restored_approval.resolution_reason == "Approved for execution."
        assert restored_approval.resolved_at is not None

        restored_run = run_repository.get(run_id)
        assert restored_run is not None
        assert restored_run.status is AgentRunStatus.COMPLETED
        assert restored_run.output == "Notification sent after approval."
        assert restored_run.lease_id is None
        assert restored_run.lease_expires_at is None
        assert restored_run.completed_at is not None

        restored_step = step_repository.get(run_id, step_id)
        assert restored_step is not None
        assert restored_step.status is AgentRunStepStatus.COMPLETED
        assert restored_step.tool_name == "send_notification"
        assert restored_step.call_id == call_id
        assert restored_step.output == {"status": "notification_sent"}
        assert restored_step.completed_at is not None

        restored_checkpoint = checkpoint_repository.get_latest(run_id)
        assert restored_checkpoint is not None
        assert restored_checkpoint.run_id == run_id
        assert restored_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION

        with session_factory() as event_session:
            events = event_session.scalars(
                select(AgentRunEventRecord)
                .where(
                    AgentRunEventRecord.run_id == run_id,
                    AgentRunEventRecord.event_type
                    == AgentExecutionEventType.APPROVAL_DECISION.value,
                )
                .order_by(AgentRunEventRecord.id.asc())
            ).all()

        assert len(events) == 1
        event = events[0]
        assert event.agent_name == "approval-continuation-agent"
        assert event.session_id == request.session_id
        assert event.user_id == request.user_id
        assert event.principal == principal
        assert event.tool_name == "send_notification"
        assert event.call_id == call_id
        assert event.step_id == step_id
        assert event.event_metadata == {
            "approval_id": approval_id,
            "decision": "approved",
            "actor": "approver-integration",
            "reason": "Approved for execution.",
        }

    finally:
        session.rollback()
        session.execute(
            delete(AgentRunEventRecord).where(
                AgentRunEventRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunStepRecord).where(
                AgentRunStepRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(ApprovalRequestRecord).where(
                ApprovalRequestRecord.approval_id == approval_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
        session.close()


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
