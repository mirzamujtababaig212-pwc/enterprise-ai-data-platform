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
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.approvals.continuation_service import (
    AgentRunApprovalContinuationService,
)
from app.control_plane.approvals.models import ApprovalRequest, ApprovalStatus
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


def _service(
    *,
    approval: ApprovalRequest | None = None,
    run: AgentRun | None = None,
    step: AgentRunStep | None = None,
    checkpoint: AgentExecutionCheckpoint | None = None,
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

    return (
        AgentRunApprovalContinuationService(
            runtime=runtime,
            approval_repository=approval_repository,
            agent_run_repository=agent_run_repository,
            agent_run_steps_repository=agent_run_steps_repository,
            checkpoints_repository=checkpoints_repository,
            lease_seconds=60,
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


@pytest.mark.asyncio
async def test_rejected_continuation_rejects_run_without_resuming() -> None:
    approval = _approval()
    run = _run()
    step = _step()
    checkpoint = _checkpoint()

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


@pytest.mark.asyncio
async def test_already_resolved_approval_can_continue_waiting_run_without_reresolving() -> None:
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

    await service.continue_approval(
        APPROVAL_ID,
        status=ApprovalStatus.APPROVED,
        resolved_by="approver-1",
    )

    approval_repository.update_status.assert_not_called()
    agent_run_repository.claim_waiting_for_approval.assert_called_once()
    runtime.resume.assert_awaited_once()


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
