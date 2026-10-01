"""PostgreSQL integration tests for delegated child approval lifecycle."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_messages import user_message
from ai_platform.agents.models import AgentDefinition, AgentRequest, AgentResponse
from ai_platform.agents.registry import InMemoryAgentRegistry
from app.control_plane.agent_delegation.models import AgentDelegationRequest
from app.control_plane.agent_delegation.service import AgentDelegationService
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
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)
from app.control_plane.approvals.continuation_service import (
    AgentRunApprovalContinuationService,
)
from app.control_plane.approvals.models import ApprovalRequest, ApprovalStatus
from app.control_plane.approvals.postgres_repository import (
    PostgreSQLApprovalRequestRepository,
)
from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    AgentRunCheckpointRecord,
    AgentRunRecord,
    AgentRunStepRecord,
    ApprovalRequestRecord,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


class DelegatedApprovalAgent:
    """Deterministic delegated child target used only for durable lifecycle setup."""

    def __init__(self) -> None:
        self._definition = AgentDefinition(
            name="integration-approval-child-agent",
            description="PostgreSQL delegated approval child.",
            system_prompt="Execute the approved enterprise operation.",
        )

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(self, context):
        raise AssertionError("The delegated child must be resumed through approval continuation.")


class DelegatedApprovalRuntime:
    """Continuation runtime that durably completes the child's tool step."""

    def __init__(self, *, run_id: str, step_id: str) -> None:
        self._run_id = run_id
        self._step_id = step_id
        self.calls = 0

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
        self.calls += 1
        assert run_id == self._run_id
        assert checkpoint.run_id == self._run_id
        assert checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
        assert lease_id
        completed_at = datetime.now(UTC)
        with SessionLocal() as session:
            repository = PostgreSQLAgentRunStepsRepository(session)
            completed_step = repository.transition(
                self._run_id,
                self._step_id,
                status=AgentRunStepStatus.COMPLETED,
                updated_at=completed_at,
                completed_at=completed_at,
                output={
                    "status": "approved",
                    "message": "Delegated operation completed.",
                },
            )
            assert completed_step is not None
            assert completed_step.status is AgentRunStepStatus.COMPLETED
        return AgentResponse(
            agent_name=agent_name,
            output="Delegated operation completed.",
            session_id=request.session_id,
        )


def _build_delegation_service(
    *,
    registry: InMemoryAgentRegistry,
) -> AgentDelegationService:
    return AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=PostgreSQLAgentRunRepository(SessionLocal()),
        agent_run_steps_repository_factory=lambda: (
            PostgreSQLAgentRunStepsRepository(SessionLocal())
        ),
    )


def _build_parent(
    *,
    parent_run_id: str,
    suffix: str,
) -> AgentRun:
    session_id = f"delegated-approval-session-{suffix}"
    user_id = f"delegated-approval-user-{suffix}"
    principal = f"api_key:delegated-approval-{suffix}"
    tenant_id = f"delegated-approval-tenant-{suffix}"
    request = AgentRequest(
        input="Delegate an approval-gated enterprise operation.",
        session_id=session_id,
        user_id=user_id,
        principal=principal,
        tenant_id=tenant_id,
        metadata={
            "source": "postgres-delegated-approval-integration",
        },
    )
    return AgentRun(
        run_id=parent_run_id,
        agent_name="integration-parent-agent",
        root_run_id=parent_run_id,
        session_id=session_id,
        user_id=user_id,
        principal=principal,
        tenant_id=tenant_id,
        status=AgentRunStatus.RUNNING,
        metadata={
            "integration": "delegated-approval",
        },
        request_snapshot=AgentRunRequestSnapshot.from_request(request),
    )


def _build_child_tool_step(
    *,
    child_run_id: str,
    call_id: str,
    tool_name: str,
) -> AgentRunStep:
    now = datetime.now(UTC)
    return AgentRunStep(
        run_id=child_run_id,
        step_id=f"tool-step-{uuid4().hex[:12]}",
        step_index=0,
        step_type="tool_call",
        status=AgentRunStepStatus.RUNNING,
        attempt=1,
        tool_name=tool_name,
        call_id=call_id,
        input={
            "amount": 500,
            "recipient": "delegated-recipient",
        },
        started_at=now,
        created_at=now,
        updated_at=now,
        metadata={
            "integration": "delegated-approval",
        },
    )


def _build_checkpoint(
    *,
    child_run: AgentRun,
) -> AgentExecutionCheckpoint:
    return AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id=child_run.run_id,
        agent_name=child_run.agent_name,
        session_id=child_run.session_id,
        user_id=child_run.user_id,
        messages=(
            user_message(
                "Complete the delegated approval-gated operation.",
            ),
        ),
        tool_round=0,
        position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
        metadata={
            "integration": "delegated-approval",
        },
    )


def _build_approval(
    *,
    child_run: AgentRun,
    step: AgentRunStep,
    call_id: str,
    suffix: str,
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=f"approval-{suffix[:12]}",
        run_id=child_run.run_id,
        step_id=step.step_id,
        call_id=call_id,
        tool_name="transfer_funds",
        idempotency_key=f"tool-exec-{suffix[:12]}",
        status=ApprovalStatus.PENDING,
        policy_name="side-effect-approval",
        policy_version="1.0.0",
        risk_tier="high",
        requested_action="Transfer 500 to delegated-recipient.",
        policy_metadata={
            "integration": "delegated-approval",
        },
    )


def _prepare_delegated_child(
    *,
    service: AgentDelegationService,
    parent_run_id: str,
    suffix: str,
):
    lease_id = f"lease-{suffix[:12]}"
    request = AgentDelegationRequest(
        parent_run_id=parent_run_id,
        child_agent_name="integration-approval-child-agent",
        child_request=AgentRequest(
            input="Execute the approval-gated delegated operation.",
        ),
        idempotency_key=f"del-appr-{suffix[:12]}",
        causation_id=f"del-appr-caus-{suffix[:12]}",
        metadata={
            "source": "postgres-delegated-approval",
        },
    )
    result = asyncio.run(service.delegate(request))
    assert result.created is True
    child_run_id = result.child_run.run_id
    parent_step_id = result.parent_step_id

    parent_repository = PostgreSQLAgentRunRepository(SessionLocal())
    parent_step_repository = PostgreSQLAgentRunStepsRepository(SessionLocal())

    try:
        child = parent_repository.get(child_run_id)
        assert child is not None
        assert child.status is AgentRunStatus.PENDING
        assert child.parent_run_id == parent_run_id
        assert child.parent_step_id == parent_step_id
        assert child.root_run_id == parent_run_id

        running_parent_step = parent_step_repository.transition(
            parent_run_id,
            parent_step_id,
            status=AgentRunStepStatus.RUNNING,
            updated_at=datetime.now(UTC),
            started_at=datetime.now(UTC),
        )
        assert running_parent_step is not None
        assert running_parent_step.status is AgentRunStepStatus.RUNNING

        started_child = parent_repository.claim_pending_run(
            child_run_id,
            started_at=datetime.now(UTC),
            lease_id=lease_id,
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        assert started_child is not None
        assert started_child.status is AgentRunStatus.RUNNING

        waiting_child = parent_repository.transition_to_waiting_for_approval_if_owner(
            child_run_id,
            lease_id=lease_id,
            updated_at=datetime.now(UTC),
        )
        assert waiting_child is not None
        assert waiting_child.status is AgentRunStatus.WAITING_FOR_APPROVAL

        call_id = f"call-{suffix[:12]}"
        step = _build_child_tool_step(
            child_run_id=child_run_id,
            call_id=call_id,
            tool_name="transfer_funds",
        )
        child_step_repository = PostgreSQLAgentRunStepsRepository(SessionLocal())
        try:
            persisted_step = child_step_repository.create(step)
            assert persisted_step.status is AgentRunStepStatus.RUNNING
        finally:
            child_step_repository._session.close()

        checkpoint = _build_checkpoint(child_run=waiting_child)
        checkpoint_repository = PostgreSQLAgentCheckpointsRepository(SessionLocal())
        try:
            checkpoint_repository.save(checkpoint)
        finally:
            checkpoint_repository._session.close()

        approval = _build_approval(
            child_run=waiting_child,
            step=step,
            call_id=call_id,
            suffix=suffix,
        )
        approval_repository = PostgreSQLApprovalRequestRepository(SessionLocal())
        try:
            approval_repository.create(approval)
        finally:
            approval_repository.close()

        return {
            "child_run_id": child_run_id,
            "parent_step_id": parent_step_id,
            "child_step_id": step.step_id,
            "call_id": call_id,
            "approval_id": approval.approval_id,
        }
    finally:
        parent_step_repository._session.close()
        parent_repository._session.close()


def test_postgres_delegated_child_approval_completes_parent_delegation() -> None:
    """An approved delegated child completes and reconciles its parent step."""
    suffix = uuid4().hex
    parent_run_id = str(uuid4())
    registry = InMemoryAgentRegistry()
    asyncio.run(registry.register(DelegatedApprovalAgent()))

    parent_session = SessionLocal()
    parent_repository = PostgreSQLAgentRunRepository(parent_session)
    child_run_id = None
    parent_step_id = None
    child_step_id = None
    approval_id = None

    try:
        parent = _build_parent(
            parent_run_id=parent_run_id,
            suffix=suffix,
        )
        parent_repository.create(parent)

        delegation_service = _build_delegation_service(
            registry=registry,
        )

        lifecycle = _prepare_delegated_child(
            service=delegation_service,
            parent_run_id=parent_run_id,
            suffix=suffix,
        )
        child_run_id = lifecycle["child_run_id"]
        parent_step_id = lifecycle["parent_step_id"]
        child_step_id = lifecycle["child_step_id"]
        approval_id = lifecycle["approval_id"]

        continuation_session = SessionLocal()
        try:
            continuation = AgentRunApprovalContinuationService(
                runtime=DelegatedApprovalRuntime(
                    run_id=child_run_id,
                    step_id=child_step_id,
                ),
                approval_repository=PostgreSQLApprovalRequestRepository(continuation_session),
                agent_run_repository=PostgreSQLAgentRunRepository(continuation_session),
                agent_run_steps_repository=PostgreSQLAgentRunStepsRepository(continuation_session),
                checkpoints_repository=PostgreSQLAgentCheckpointsRepository(continuation_session),
                lease_seconds=60,
                delegated_child_reconciler=delegation_service.reconcile_child_run,
            )

            response = asyncio.run(
                continuation.continue_approval(
                    approval_id,
                    status=ApprovalStatus.APPROVED,
                    resolved_by="integration-approver",
                    resolution_reason="Approved delegated operation.",
                )
            )
        finally:
            continuation_session.close()
        assert response.output == "Delegated operation completed."

        with SessionLocal() as verify_session:
            approval_repository = PostgreSQLApprovalRequestRepository(
                verify_session,
            )
            run_repository = PostgreSQLAgentRunRepository(
                verify_session,
            )
            step_repository = PostgreSQLAgentRunStepsRepository(
                verify_session,
            )

            restored_approval = approval_repository.get(approval_id)
            assert restored_approval is not None
            assert restored_approval.status is ApprovalStatus.APPROVED
            assert restored_approval.resolved_by == "integration-approver"
            assert restored_approval.resolution_reason == "Approved delegated operation."

            child = run_repository.get(child_run_id)
            assert child is not None
            assert child.status is AgentRunStatus.COMPLETED
            assert child.output == "Delegated operation completed."
            assert child.parent_run_id == parent_run_id
            assert child.parent_step_id == parent_step_id
            assert child.root_run_id == parent_run_id
            assert child.lease_id is None
            assert child.lease_expires_at is None

            child_step = step_repository.get(
                child_run_id,
                child_step_id,
            )
            assert child_step is not None
            assert child_step.status is AgentRunStepStatus.COMPLETED
            assert child_step.output == {
                "status": "approved",
                "message": "Delegated operation completed.",
            }

            parent_step = step_repository.get(
                parent_run_id,
                parent_step_id,
            )
            assert parent_step is not None
            assert parent_step.status is AgentRunStepStatus.COMPLETED
            assert parent_step.output == "Delegated operation completed."

            parent_restored = run_repository.get(parent_run_id)
            assert parent_restored is not None
            assert parent_restored.status is AgentRunStatus.RUNNING

            children = [
                run
                for run in run_repository.list(
                    tenant_id=parent_restored.tenant_id,
                )
                if run.parent_run_id == parent_run_id
            ]
            assert [run.run_id for run in children] == [child_run_id]

    finally:
        cleanup_session = SessionLocal()
        try:
            if approval_id is not None:
                cleanup_session.execute(
                    delete(ApprovalRequestRecord).where(
                        ApprovalRequestRecord.approval_id == approval_id,
                    )
                )
            if child_run_id is not None:
                cleanup_session.execute(
                    delete(AgentRunCheckpointRecord).where(
                        AgentRunCheckpointRecord.run_id == child_run_id,
                    )
                )
            if child_step_id is not None and child_run_id is not None:
                cleanup_session.execute(
                    delete(AgentRunStepRecord).where(
                        AgentRunStepRecord.run_id == child_run_id,
                        AgentRunStepRecord.step_id == child_step_id,
                    )
                )
            if child_run_id is not None:
                cleanup_session.execute(
                    delete(AgentRunRecord).where(
                        AgentRunRecord.run_id == child_run_id,
                    )
                )
            # Delete parent steps and parent run after child run deletion to satisfy FK constraints
            if parent_step_id is not None:
                cleanup_session.execute(
                    delete(AgentRunStepRecord).where(
                        AgentRunStepRecord.run_id == parent_run_id,
                        AgentRunStepRecord.step_id == parent_step_id,
                    )
                )
            cleanup_session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.run_id == parent_run_id,
                )
            )
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            parent_session.close()


def test_postgres_delegated_child_rejection_fails_parent_delegation() -> None:
    """A rejected delegated child fails its parent delegation step."""
    suffix = uuid4().hex
    parent_run_id = str(uuid4())
    registry = InMemoryAgentRegistry()
    asyncio.run(registry.register(DelegatedApprovalAgent()))

    parent_session = SessionLocal()
    parent_repository = PostgreSQLAgentRunRepository(parent_session)
    child_run_id = None
    parent_step_id = None
    child_step_id = None
    approval_id = None

    try:
        parent = _build_parent(
            parent_run_id=parent_run_id,
            suffix=suffix,
        )
        parent_repository.create(parent)

        delegation_service = _build_delegation_service(
            registry=registry,
        )

        lifecycle = _prepare_delegated_child(
            service=delegation_service,
            parent_run_id=parent_run_id,
            suffix=suffix,
        )
        child_run_id = lifecycle["child_run_id"]
        parent_step_id = lifecycle["parent_step_id"]
        child_step_id = lifecycle["child_step_id"]
        approval_id = lifecycle["approval_id"]

        continuation_session = SessionLocal()
        try:
            continuation = AgentRunApprovalContinuationService(
                runtime=DelegatedApprovalRuntime(
                    run_id=child_run_id,
                    step_id=child_step_id,
                ),
                approval_repository=PostgreSQLApprovalRequestRepository(continuation_session),
                agent_run_repository=PostgreSQLAgentRunRepository(continuation_session),
                agent_run_steps_repository=PostgreSQLAgentRunStepsRepository(continuation_session),
                checkpoints_repository=PostgreSQLAgentCheckpointsRepository(continuation_session),
                lease_seconds=60,
                delegated_child_reconciler=delegation_service.reconcile_child_run,
            )

            with pytest.raises(RuntimeError, match="Delegated operation was rejected."):
                asyncio.run(
                    continuation.continue_approval(
                        approval_id,
                        status=ApprovalStatus.REJECTED,
                        resolved_by="integration-approver",
                        resolution_reason="Delegated operation was rejected.",
                    )
                )
        finally:
            continuation_session.close()

        with SessionLocal() as verify_session:
            approval_repository = PostgreSQLApprovalRequestRepository(
                verify_session,
            )
            run_repository = PostgreSQLAgentRunRepository(
                verify_session,
            )
            step_repository = PostgreSQLAgentRunStepsRepository(
                verify_session,
            )

            restored_approval = approval_repository.get(approval_id)
            assert restored_approval is not None
            assert restored_approval.status is ApprovalStatus.REJECTED
            assert restored_approval.resolved_by == "integration-approver"
            assert restored_approval.resolution_reason == "Delegated operation was rejected."

            child = run_repository.get(child_run_id)
            assert child is not None
            assert child.status is AgentRunStatus.REJECTED
            assert child.error_type == "ApprovalRejected"
            assert child.error_message == "Delegated operation was rejected."
            assert child.parent_run_id == parent_run_id
            assert child.parent_step_id == parent_step_id
            assert child.root_run_id == parent_run_id
            assert child.lease_id is None
            assert child.lease_expires_at is None

            child_step = step_repository.get(
                child_run_id,
                child_step_id,
            )
            assert child_step is not None
            assert child_step.status is AgentRunStepStatus.FAILED
            assert child_step.error == "Delegated operation was rejected."
            assert child_step.failure_category == "approval_rejected"

            parent_step = step_repository.get(
                parent_run_id,
                parent_step_id,
            )
            assert parent_step is not None
            assert parent_step.status is AgentRunStepStatus.FAILED
            assert parent_step.error == "Delegated operation was rejected."
            assert parent_step.failure_category == "ApprovalRejected"

            parent_restored = run_repository.get(parent_run_id)
            assert parent_restored is not None
            assert parent_restored.status is AgentRunStatus.RUNNING

    finally:
        cleanup_session = SessionLocal()
        try:
            if approval_id is not None:
                cleanup_session.execute(
                    delete(ApprovalRequestRecord).where(
                        ApprovalRequestRecord.approval_id == approval_id,
                    )
                )
            if child_run_id is not None:
                cleanup_session.execute(
                    delete(AgentRunCheckpointRecord).where(
                        AgentRunCheckpointRecord.run_id == child_run_id,
                    )
                )
            if child_step_id is not None and child_run_id is not None:
                cleanup_session.execute(
                    delete(AgentRunStepRecord).where(
                        AgentRunStepRecord.run_id == child_run_id,
                        AgentRunStepRecord.step_id == child_step_id,
                    )
                )
            if child_run_id is not None:
                cleanup_session.execute(
                    delete(AgentRunRecord).where(
                        AgentRunRecord.run_id == child_run_id,
                    )
                )
            if parent_step_id is not None:
                cleanup_session.execute(
                    delete(AgentRunStepRecord).where(
                        AgentRunStepRecord.run_id == parent_run_id,
                        AgentRunStepRecord.step_id == parent_step_id,
                    )
                )
            cleanup_session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.run_id == parent_run_id,
                )
            )
            cleanup_session.commit()
        finally:
            cleanup_session.close()
            parent_session.close()
