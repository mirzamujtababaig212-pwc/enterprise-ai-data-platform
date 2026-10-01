from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.exceptions import AgentExecutionControlSignal
from ai_platform.agents.models import AgentRequest, AgentResponse
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.approvals.continuation_service import (
    AgentRunApprovalContinuationService,
)
from app.control_plane.approvals.authorization import (
    ConfiguredApprovalOverrideAuthorizer,
)
from app.control_plane.approvals.models import (
    ApprovalOverride,
    ApprovalRequest,
    ApprovalStatus,
)
from ai_platform.agents.llm_messages import user_message

RUN_ID = "run-1"
STEP_ID = "step-1"
CALL_ID = "call-1"
APPROVAL_ID = "approval-1"
TOOL_NAME = "send_payment"


def _request() -> AgentRequest:
    return AgentRequest(
        input="Send payment of 100",
        session_id="session-1",
        user_id="user-1",
        memory_namespace="tenant-memory",
        metadata={"request_id": "request-1"},
    )


def _snapshot() -> AgentRunRequestSnapshot:
    return AgentRunRequestSnapshot.from_request(_request())


def _run(
    *,
    status: AgentRunStatus = AgentRunStatus.WAITING_FOR_APPROVAL,
) -> AgentRun:
    return AgentRun(
        run_id=RUN_ID,
        agent_name="payment-agent",
        session_id="session-1",
        user_id="user-1",
        principal="api_key:test-principal",
        tenant_id="tenant-1",
        idempotency_key=None,
        status=status,
        started_at=datetime.now(UTC),
        completed_at=None,
        lease_id=None,
        lease_expires_at=None,
        error_type=None,
        error_message=None,
        output=None,
        metadata={"request_id": "request-1"},
        request_snapshot=_snapshot(),
        recovery_attempts=0,
    )


def _step() -> AgentRunStep:
    return AgentRunStep(
        step_id=STEP_ID,
        run_id=RUN_ID,
        step_index=0,
        step_type="tool_execution",
        status=AgentRunStepStatus.RUNNING,
        attempt=1,
        started_at=datetime.now(UTC),
        completed_at=None,
        call_id=CALL_ID,
        tool_name=TOOL_NAME,
        input={"amount": 100},
        output=None,
        error=None,
        failure_category=None,
    )


def _approval(
    *,
    status: ApprovalStatus = ApprovalStatus.PENDING,
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=APPROVAL_ID,
        run_id=RUN_ID,
        step_id=STEP_ID,
        call_id=CALL_ID,
        tool_name=TOOL_NAME,
        idempotency_key="idem-1",
        status=status,
        policy_name="high-risk-tool-approval",
        policy_version="1.0.0",
        risk_tier="high",
        requested_action="Execute payment",
        policy_metadata={"source": "tenant-policy"},
    )


def _checkpoint(
    *,
    position: AgentCheckpointPosition = AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
    run_id: str = RUN_ID,
) -> AgentExecutionCheckpoint:
    return AgentExecutionCheckpoint(
        schema_version=1,
        run_id=run_id,
        agent_name="payment-agent",
        session_id="session-1",
        user_id="user-1",
        messages=(user_message("Send payment of 100"),),
        tool_round=1,
        position=position,
        metadata={},
    )


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(self, event: AgentExecutionEvent) -> None:
        self.events.append(event)


class FailingObserver:
    async def record(self, event: AgentExecutionEvent) -> None:
        raise RuntimeError("observer failure")


def _service(
    *,
    approval: ApprovalRequest | None = None,
    run: AgentRun | None = None,
    step: AgentRunStep | None = None,
    checkpoint: AgentExecutionCheckpoint | None = None,
    override_repository: MagicMock | None = None,
    override_authorizer: ConfiguredApprovalOverrideAuthorizer | None = None,
    observer: object | None = None,
    delegated_child_reconciler: MagicMock | None = None,
):
    runtime = MagicMock()
    runtime.resume = AsyncMock(
        return_value=AgentResponse(
            agent_name="payment-agent",
            output="Payment completed",
        )
    )

    approval_repository = MagicMock()
    approval_repository.get.return_value = approval
    approval_repository.update_status.side_effect = (
        lambda approval_id, status, **kwargs: approval.model_copy(
            update={
                "status": status,
                "resolved_by": kwargs.get("resolved_by"),
                "resolution_reason": kwargs.get("resolution_reason"),
                "resolved_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
        )
    )

    agent_run_repository = MagicMock()
    agent_run_repository.get.return_value = run

    def claim_waiting_for_approval(
        run_id: str,
        *,
        lease_id: str,
        lease_expires_at: datetime,
    ) -> AgentRun:
        return run.model_copy(
            update={
                "status": AgentRunStatus.RUNNING,
                "lease_id": lease_id,
                "lease_expires_at": lease_expires_at,
            }
        )

    agent_run_repository.claim_waiting_for_approval.side_effect = claim_waiting_for_approval
    agent_run_repository.complete_if_owner.side_effect = lambda run_id, **kwargs: run.model_copy(
        update={
            "status": AgentRunStatus.COMPLETED,
            "completed_at": kwargs["completed_at"],
            "output": kwargs["output"],
            "lease_id": None,
            "lease_expires_at": None,
        }
    )

    agent_run_steps_repository = MagicMock()
    agent_run_steps_repository.get.return_value = step

    checkpoints_repository = MagicMock()
    checkpoints_repository.get_latest.return_value = checkpoint

    if override_repository is None:
        override_repository = MagicMock()
        override_repository.get_by_approval.return_value = None

    if override_authorizer is None:
        override_authorizer = ConfiguredApprovalOverrideAuthorizer(
            frozenset({"api_key:operator-1"}),
        )

    return (
        AgentRunApprovalContinuationService(
            runtime=runtime,
            approval_repository=approval_repository,
            agent_run_repository=agent_run_repository,
            agent_run_steps_repository=agent_run_steps_repository,
            checkpoints_repository=checkpoints_repository,
            override_repository=override_repository,
            lease_seconds=60,
            override_authorizer=override_authorizer,
            observer=observer,
            delegated_child_reconciler=delegated_child_reconciler,
        ),
        runtime,
        approval_repository,
        agent_run_repository,
        agent_run_steps_repository,
        checkpoints_repository,
    )


@pytest.mark.asyncio
async def test_approved_continuation_resumes_and_completes_run() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()
    observer = RecordingObserver()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        observer=observer,
    )

    response = await service.continue_approval(
        APPROVAL_ID,
        status=ApprovalStatus.APPROVED,
        resolved_by="approver-1",
        resolution_reason="Approved for execution",
    )

    assert response.output == "Payment completed"

    approval_repository.update_status.assert_called_once_with(
        APPROVAL_ID,
        ApprovalStatus.APPROVED,
        resolved_by="approver-1",
        resolution_reason="Approved for execution",
        commit=True,
    )

    agent_run_repository.claim_waiting_for_approval.assert_called_once()

    runtime.resume.assert_awaited_once()
    resume_args = runtime.resume.await_args.args
    resume_kwargs = runtime.resume.await_args.kwargs
    assert resume_args[0] == "payment-agent"
    assert resume_args[2] is checkpoint
    assert resume_kwargs["run_id"] == RUN_ID
    assert resume_kwargs["lease_id"]

    agent_run_repository.complete_if_owner.assert_called_once()

    assert len(observer.events) == 1
    event = observer.events[0]
    assert event.event_type is AgentExecutionEventType.APPROVAL_DECISION
    assert event.agent_name == run.agent_name
    assert event.run_id == RUN_ID
    assert event.session_id == run.session_id
    assert event.user_id == run.user_id
    assert event.tool_name == approval.tool_name
    assert event.call_id == approval.call_id
    assert event.step_id == approval.step_id
    assert event.metadata == {
        "approval_id": APPROVAL_ID,
        "decision": "approved",
        "actor": "approver-1",
        "reason": "Approved for execution",
    }


@pytest.mark.asyncio
async def test_approved_delegated_child_reconciles_parent_step() -> None:
    approval = _approval()
    run = _run().model_copy(
        update={
            "parent_run_id": "parent-run-1",
            "parent_step_id": "delegation-step-1",
        }
    )
    step = _step()
    checkpoint = _checkpoint()
    reconciler = MagicMock()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        delegated_child_reconciler=reconciler,
    )

    response = await service.continue_approval(
        APPROVAL_ID,
        status=ApprovalStatus.APPROVED,
        resolved_by="approver-1",
        resolution_reason="Approved for execution",
    )

    assert response.output == "Payment completed"
    runtime.resume.assert_awaited_once()
    agent_run_repository.complete_if_owner.assert_called_once()

    reconciler.assert_called_once()

    reconciler_kwargs = reconciler.call_args.kwargs
    assert reconciler_kwargs["parent_run_id"] == "parent-run-1"
    assert reconciler_kwargs["parent_step_id"] == "delegation-step-1"

    completed_child = reconciler_kwargs["child_run"]
    assert completed_child.status is AgentRunStatus.COMPLETED
    assert completed_child.output == "Payment completed"
    assert completed_child.parent_run_id == "parent-run-1"
    assert completed_child.parent_step_id == "delegation-step-1"


@pytest.mark.asyncio
async def test_approval_observer_failure_does_not_block_approved_continuation() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        observer=FailingObserver(),
    )

    response = await service.continue_approval(
        APPROVAL_ID,
        status=ApprovalStatus.APPROVED,
        resolved_by="approver-1",
        resolution_reason="Approved for execution",
    )

    assert response.output == "Payment completed"
    approval_repository.update_status.assert_called_once_with(
        APPROVAL_ID,
        ApprovalStatus.APPROVED,
        resolved_by="approver-1",
        resolution_reason="Approved for execution",
        commit=True,
    )
    agent_run_repository.claim_waiting_for_approval.assert_called_once()
    runtime.resume.assert_awaited_once()
    agent_run_repository.complete_if_owner.assert_called_once()


@pytest.mark.asyncio
async def test_rejected_continuation_rejects_run_without_resuming() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()
    observer = RecordingObserver()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        agent_run_steps_repository,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        observer=observer,
    )

    agent_run_repository.reject_waiting_for_approval.return_value = run.model_copy(
        update={
            "status": AgentRunStatus.REJECTED,
            "completed_at": datetime.now(UTC),
            "error_type": "ApprovalRejected",
            "error_message": "Payment not authorized",
        }
    )

    with pytest.raises(RuntimeError, match="Payment not authorized"):
        await service.continue_approval(
            APPROVAL_ID,
            status=ApprovalStatus.REJECTED,
            resolved_by="approver-1",
            resolution_reason="Payment not authorized",
        )

    approval_repository.update_status.assert_called_once_with(
        APPROVAL_ID,
        ApprovalStatus.REJECTED,
        resolved_by="approver-1",
        resolution_reason="Payment not authorized",
        commit=False,
    )

    agent_run_steps_repository.transition.assert_called_once()

    transition_kwargs = agent_run_steps_repository.transition.call_args.kwargs

    assert transition_kwargs["status"] is AgentRunStepStatus.FAILED
    assert transition_kwargs["updated_at"] == transition_kwargs["completed_at"]
    assert transition_kwargs["error"] == "Payment not authorized"
    assert transition_kwargs["failure_category"] == "approval_rejected"
    assert transition_kwargs["commit"] is False

    agent_run_repository.reject_waiting_for_approval.assert_called_once()

    rejection_kwargs = agent_run_repository.reject_waiting_for_approval.call_args.kwargs

    assert rejection_kwargs["error_type"] == "ApprovalRejected"
    assert rejection_kwargs["error_message"] == "Payment not authorized"
    assert rejection_kwargs["commit"] is True

    agent_run_repository.claim_waiting_for_approval.assert_not_called()
    runtime.resume.assert_not_awaited()
    agent_run_repository.complete_if_owner.assert_not_called()

    assert len(observer.events) == 1
    event = observer.events[0]
    assert event.event_type is AgentExecutionEventType.APPROVAL_DECISION
    assert event.agent_name == run.agent_name
    assert event.run_id == RUN_ID
    assert event.session_id == run.session_id
    assert event.user_id == run.user_id
    assert event.tool_name == approval.tool_name
    assert event.call_id == approval.call_id
    assert event.step_id == approval.step_id
    assert event.metadata == {
        "approval_id": APPROVAL_ID,
        "decision": "rejected",
        "actor": "approver-1",
        "reason": "Payment not authorized",
    }


@pytest.mark.asyncio
async def test_rejected_delegated_child_reconciles_parent_step() -> None:
    approval = _approval()
    run = _run().model_copy(
        update={
            "parent_run_id": "parent-run-1",
            "parent_step_id": "delegation-step-1",
        }
    )
    step = _step()
    checkpoint = _checkpoint()
    reconciler = MagicMock()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        delegated_child_reconciler=reconciler,
    )

    rejected_run = run.model_copy(
        update={
            "status": AgentRunStatus.REJECTED,
            "completed_at": datetime.now(UTC),
            "error_type": "ApprovalRejected",
            "error_message": "Payment not authorized",
        }
    )
    agent_run_repository.reject_waiting_for_approval.return_value = rejected_run

    with pytest.raises(RuntimeError, match="Payment not authorized"):
        await service.continue_approval(
            APPROVAL_ID,
            status=ApprovalStatus.REJECTED,
            resolved_by="approver-1",
            resolution_reason="Payment not authorized",
        )

    reconciler.assert_called_once_with(
        parent_run_id="parent-run-1",
        parent_step_id="delegation-step-1",
        child_run=rejected_run,
    )

    runtime.resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_delegated_approval_does_not_invoke_reconciler() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()
    reconciler = MagicMock()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        delegated_child_reconciler=reconciler,
    )

    await service.continue_approval(
        APPROVAL_ID,
        status=ApprovalStatus.APPROVED,
        resolved_by="approver-1",
    )

    runtime.resume.assert_awaited_once()
    agent_run_repository.complete_if_owner.assert_called_once()
    reconciler.assert_not_called()


@pytest.mark.asyncio
async def test_already_resolved_approval_can_continue_waiting_run_without_reresolving() -> None:
    approval = _approval(status=ApprovalStatus.APPROVED)
    run = _run()
    step = _step()
    checkpoint = _checkpoint()
    observer = RecordingObserver()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        observer=observer,
    )

    await service.continue_approval(
        APPROVAL_ID,
        status=ApprovalStatus.APPROVED,
        resolved_by="approver-1",
    )

    approval_repository.update_status.assert_not_called()
    agent_run_repository.claim_waiting_for_approval.assert_called_once()
    runtime.resume.assert_awaited_once()
    assert observer.events == []


@pytest.mark.asyncio
async def test_second_continuation_cannot_claim_run_after_first_claim() -> None:
    approval = _approval(status=ApprovalStatus.APPROVED)
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
    )

    agent_run_repository.claim_waiting_for_approval.side_effect = None
    agent_run_repository.claim_waiting_for_approval.return_value = None

    with pytest.raises(RuntimeError, match="could not be claimed"):
        await service.continue_approval(
            APPROVAL_ID,
            status=ApprovalStatus.APPROVED,
            resolved_by="approver-1",
        )

    runtime.resume.assert_not_awaited()
    agent_run_repository.complete_if_owner.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_checkpoint_is_rejected_before_approval_resolution_and_claim() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint(
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
    )

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match="requires a BEFORE_TOOL_EXECUTION checkpoint",
    ):
        await service.continue_approval(
            APPROVAL_ID,
            status=ApprovalStatus.APPROVED,
            resolved_by="approver-1",
        )

    approval_repository.update_status.assert_not_called()
    agent_run_repository.claim_waiting_for_approval.assert_not_called()
    runtime.resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_control_signal_returns_run_to_waiting_for_approval() -> None:
    approval = _approval(status=ApprovalStatus.APPROVED)
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
    )

    runtime.resume.side_effect = AgentExecutionControlSignal("approval required again")

    agent_run_repository.transition_to_waiting_for_approval_if_owner.return_value = run

    with pytest.raises(AgentExecutionControlSignal):
        await service.continue_approval(
            APPROVAL_ID,
            status=ApprovalStatus.APPROVED,
            resolved_by="approver-1",
        )

    agent_run_repository.transition_to_waiting_for_approval_if_owner.assert_called_once()
    agent_run_repository.complete_if_owner.assert_not_called()


@pytest.mark.asyncio
async def test_authorized_override_persists_and_resumes_without_approving_request() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()
    observer = RecordingObserver()

    override_repository = MagicMock()
    override_repository.get_by_approval.return_value = None

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        override_repository=override_repository,
        observer=observer,
    )

    response = await service.override_approval(
        APPROVAL_ID,
        actor="api_key:operator-1",
        reason="Emergency operational bypass.",
    )

    assert response.output == "Payment completed"

    approval_repository.update_status.assert_not_called()

    override_repository.create.assert_called_once()
    override = override_repository.create.call_args.args[0]

    assert isinstance(override, ApprovalOverride)
    assert override.approval_id == APPROVAL_ID
    assert override.run_id == RUN_ID
    assert override.actor == "api_key:operator-1"
    assert override.reason == "Emergency operational bypass."

    agent_run_repository.claim_waiting_for_approval.assert_called_once()
    runtime.resume.assert_awaited_once()
    agent_run_repository.complete_if_owner.assert_called_once()

    assert len(observer.events) == 1
    event = observer.events[0]
    assert event.event_type is AgentExecutionEventType.APPROVAL_DECISION
    assert event.agent_name == run.agent_name
    assert event.run_id == RUN_ID
    assert event.session_id == run.session_id
    assert event.user_id == run.user_id
    assert event.tool_name == approval.tool_name
    assert event.call_id == approval.call_id
    assert event.step_id == approval.step_id
    assert event.metadata == {
        "approval_id": APPROVAL_ID,
        "decision": "overridden",
        "actor": "api_key:operator-1",
        "reason": "Emergency operational bypass.",
    }


@pytest.mark.asyncio
async def test_authorized_override_of_delegated_child_reconciles_parent_step() -> None:
    approval = _approval()
    run = _run().model_copy(
        update={
            "parent_run_id": "parent-run-1",
            "parent_step_id": "delegation-step-1",
        }
    )
    step = _step()
    checkpoint = _checkpoint()
    reconciler = MagicMock()

    override_repository = MagicMock()
    override_repository.get_by_approval.return_value = None

    (
        service,
        runtime,
        approval_repository,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        override_repository=override_repository,
        delegated_child_reconciler=reconciler,
    )

    response = await service.override_approval(
        APPROVAL_ID,
        actor="api_key:operator-1",
        reason="Emergency operational bypass.",
    )

    assert response.output == "Payment completed"

    approval_repository.update_status.assert_not_called()
    override_repository.create.assert_called_once()
    agent_run_repository.claim_waiting_for_approval.assert_called_once()
    runtime.resume.assert_awaited_once()
    agent_run_repository.complete_if_owner.assert_called_once()

    reconciler.assert_called_once()

    reconciler_kwargs = reconciler.call_args.kwargs
    assert reconciler_kwargs["parent_run_id"] == "parent-run-1"
    assert reconciler_kwargs["parent_step_id"] == "delegation-step-1"

    completed_child = reconciler_kwargs["child_run"]
    assert completed_child.status is AgentRunStatus.COMPLETED
    assert completed_child.output == "Payment completed"
    assert completed_child.parent_run_id == "parent-run-1"
    assert completed_child.parent_step_id == "delegation-step-1"


@pytest.mark.asyncio
async def test_unauthorized_override_does_not_resume_or_persist() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    override_repository = MagicMock()
    override_repository.get_by_approval.return_value = None

    (
        service,
        runtime,
        _,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        override_repository=override_repository,
        override_authorizer=ConfiguredApprovalOverrideAuthorizer(
            frozenset({"api_key:operator-1"}),
        ),
    )

    with pytest.raises(PermissionError, match="not authorized"):
        await service.override_approval(
            APPROVAL_ID,
            actor="api_key:unauthorized",
            reason="Emergency operational bypass.",
        )

    override_repository.create.assert_not_called()
    agent_run_repository.claim_waiting_for_approval.assert_not_called()
    runtime.resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_override_cannot_be_reused() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    existing_override = ApprovalOverride(
        override_id="override-existing",
        approval_id=APPROVAL_ID,
        run_id=RUN_ID,
        actor="api_key:operator-1",
        reason="Existing operational bypass.",
    )

    override_repository = MagicMock()
    override_repository.get_by_approval.return_value = existing_override

    (
        service,
        runtime,
        _,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        override_repository=override_repository,
    )

    with pytest.raises(ValueError, match="already has an override"):
        await service.override_approval(
            APPROVAL_ID,
            actor="api_key:operator-1",
            reason="Second bypass.",
        )

    override_repository.create.assert_not_called()
    agent_run_repository.claim_waiting_for_approval.assert_not_called()
    runtime.resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_override_requires_authorizer_configuration() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    override_repository = MagicMock()
    override_repository.get_by_approval.return_value = None

    (
        service,
        runtime,
        _,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        override_repository=override_repository,
        override_authorizer=None,
    )

    service._override_authorizer = None

    with pytest.raises(
        RuntimeError,
        match="override authorizer is not configured",
    ):
        await service.override_approval(
            APPROVAL_ID,
            actor="api_key:operator-1",
            reason="Emergency operational bypass.",
        )

    override_repository.create.assert_not_called()
    agent_run_repository.claim_waiting_for_approval.assert_not_called()
    runtime.resume.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED],
)
async def test_resolved_approval_cannot_be_overridden(
    status: ApprovalStatus,
) -> None:
    approval = _approval(status=status)
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    override_repository = MagicMock()
    override_repository.get_by_approval.return_value = None

    (
        service,
        runtime,
        _,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        override_repository=override_repository,
    )

    with pytest.raises(ValueError, match="must be pending"):
        await service.override_approval(
            APPROVAL_ID,
            actor="api_key:operator-1",
            reason="Emergency operational bypass.",
        )

    override_repository.create.assert_not_called()
    agent_run_repository.claim_waiting_for_approval.assert_not_called()
    runtime.resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_override_requires_non_empty_actor_and_reason() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    override_repository = MagicMock()
    override_repository.get_by_approval.return_value = None

    (
        service,
        runtime,
        _,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        override_repository=override_repository,
    )

    with pytest.raises(ValueError, match="actor must not be empty"):
        await service.override_approval(
            APPROVAL_ID,
            actor="   ",
            reason="Emergency operational bypass.",
        )

    with pytest.raises(ValueError, match="reason must not be empty"):
        await service.override_approval(
            APPROVAL_ID,
            actor="api_key:operator-1",
            reason="   ",
        )

    override_repository.create.assert_not_called()
    agent_run_repository.claim_waiting_for_approval.assert_not_called()
    runtime.resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_approval_cannot_continue_after_override() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

    existing_override = ApprovalOverride(
        override_id="override-existing",
        approval_id=APPROVAL_ID,
        run_id=RUN_ID,
        actor="api_key:operator-1",
        reason="Emergency operational bypass.",
    )

    override_repository = MagicMock()
    override_repository.get_by_approval.return_value = existing_override

    (
        service,
        runtime,
        _,
        agent_run_repository,
        _,
        _,
    ) = _service(
        approval=approval,
        run=run,
        step=step,
        checkpoint=checkpoint,
        override_repository=override_repository,
    )

    with pytest.raises(
        ValueError,
        match="already has an override.*cannot be resolved normally",
    ):
        await service.continue_approval(
            APPROVAL_ID,
            status=ApprovalStatus.APPROVED,
            resolved_by="api_key:approver-1",
            resolution_reason="Approved after review.",
        )

    agent_run_repository.get.assert_not_called()
    runtime.resume.assert_not_awaited()
    override_repository.get_by_approval.assert_called_once_with(APPROVAL_ID)
