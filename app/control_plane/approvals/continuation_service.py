from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.exceptions import (
    AgentExecutionControlSignal,
    AgentExecutionOwnershipLostError,
)
from ai_platform.agents.models import AgentResponse
from ai_platform.agents.runtime import AgentRuntime
from app.control_plane.agent_runs.cancellation import (
    AgentRunCancellationRegistry,
)
from app.control_plane.agent_runs.lease import (
    create_lease,
    heartbeat_loop,
)
from app.control_plane.agent_runs.models import AgentRunStatus
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_run_steps.models import AgentRunStepStatus
from app.control_plane.agent_run_steps.repository import AgentRunStepsRepository
from app.control_plane.agent_checkpoints.repository import (
    AgentCheckpointsRepository,
)

from .models import ApprovalStatus
from .repository import ApprovalRequestRepository


class AgentRunApprovalContinuationService:
    """
    Resume a durable agent run after a human approval decision.

    Approval continuation is intentionally separate from crash recovery.
    The run is resumed from its existing BEFORE_TOOL_EXECUTION checkpoint
    after the waiting run is claimed with a fresh lease.
    """

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        approval_repository: ApprovalRequestRepository,
        agent_run_repository: AgentRunRepository,
        agent_run_steps_repository: AgentRunStepsRepository,
        checkpoints_repository: AgentCheckpointsRepository,
        cancellation_registry: AgentRunCancellationRegistry | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self._runtime = runtime
        self._approval_repository = approval_repository
        self._agent_run_repository = agent_run_repository
        self._agent_run_steps_repository = agent_run_steps_repository
        self._checkpoints_repository = checkpoints_repository
        self._cancellation_registry = cancellation_registry
        self._lease_seconds = lease_seconds

    async def continue_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        resolved_by: str,
        resolution_reason: str | None = None,
    ) -> AgentResponse:
        approval = self._approval_repository.get(approval_id)

        if approval is None:
            raise ValueError(f"Approval request '{approval_id}' was not found.")

        run = self._agent_run_repository.get(approval.run_id)

        if run is None:
            raise ValueError(
                f"Agent run '{approval.run_id}' for approval " f"'{approval_id}' was not found."
            )

        if run.status is not AgentRunStatus.WAITING_FOR_APPROVAL:
            raise ValueError(
                f"Agent run '{run.run_id}' must be waiting for approval; "
                f"current status is {run.status.value}."
            )

        step = self._agent_run_steps_repository.get(
            approval.run_id,
            approval.step_id,
        )

        if step is None:
            raise ValueError(
                f"Agent run step '{approval.step_id}' for run "
                f"'{approval.run_id}' was not found."
            )

        if step.run_id != approval.run_id:
            raise ValueError("Approval request run_id does not match the agent run step.")

        if step.step_id != approval.step_id:
            raise ValueError("Approval request step_id does not match the agent run step.")

        if step.call_id != approval.call_id:
            raise ValueError("Approval request call_id does not match the agent run step.")

        if step.tool_name != approval.tool_name:
            raise ValueError("Approval request tool_name does not match the agent run step.")

        if step.status is not AgentRunStepStatus.RUNNING:
            raise ValueError(
                f"Agent run step '{step.step_id}' must be running for approval "
                f"continuation; current status is {step.status.value}."
            )

        if approval.status is not ApprovalStatus.PENDING and approval.status is not status:
            raise ValueError(
                f"Approval request '{approval_id}' is already "
                f"{approval.status.value} and cannot be resolved as "
                f"{status.value}."
            )

        if status is ApprovalStatus.REJECTED:
            if approval.status is ApprovalStatus.PENDING:
                approval = self._approval_repository.update_status(
                    approval_id,
                    status,
                    resolved_by=resolved_by,
                    resolution_reason=resolution_reason,
                    commit=False,
                )

            rejected_at = datetime.now(UTC)
            rejection_reason = resolution_reason or "Approval request was rejected."

            step_result = self._agent_run_steps_repository.transition(
                approval.run_id,
                approval.step_id,
                status=AgentRunStepStatus.FAILED,
                updated_at=rejected_at,
                completed_at=rejected_at,
                error=rejection_reason,
                failure_category="approval_rejected",
                commit=False,
            )

            if step_result is None:
                raise RuntimeError(
                    f"Agent run step '{approval.step_id}' could not be "
                    "marked as failed after approval rejection."
                )

            rejected_run = self._agent_run_repository.reject_waiting_for_approval(
                approval.run_id,
                completed_at=rejected_at,
                error_type="ApprovalRejected",
                error_message=rejection_reason,
                commit=True,
            )

            if rejected_run is None:
                raise RuntimeError(
                    f"Agent run '{approval.run_id}' could not be rejected "
                    "after approval rejection."
                )

            raise RuntimeError(rejection_reason)

        checkpoint = self._checkpoints_repository.get_latest(approval.run_id)

        if checkpoint is None:
            raise ValueError(
                f"Agent run '{approval.run_id}' has no execution checkpoint "
                "for approval continuation."
            )

        self._validate_checkpoint(
            checkpoint,
            approval_run_id=approval.run_id,
        )

        request_snapshot = run.request_snapshot

        if request_snapshot is None:
            raise ValueError(
                f"Agent run '{approval.run_id}' has no request snapshot "
                "for approval continuation."
            )

        if approval.status is ApprovalStatus.PENDING:
            approval = self._approval_repository.update_status(
                approval_id,
                status,
                resolved_by=resolved_by,
                resolution_reason=resolution_reason,
                commit=True,
            )

        lease_id, lease_expires_at = create_lease(self._lease_seconds)

        claimed_run = self._agent_run_repository.claim_waiting_for_approval(
            approval.run_id,
            lease_id=lease_id,
            lease_expires_at=lease_expires_at,
        )

        if claimed_run is None:
            raise RuntimeError(
                f"Agent run '{approval.run_id}' could not be claimed for " "approval continuation."
            )

        request = request_snapshot.to_request(
            session_id=claimed_run.session_id,
            user_id=claimed_run.user_id,
            principal=claimed_run.principal,
        )

        ownership_lost = asyncio.Event()
        execution_task = asyncio.current_task()

        if execution_task is not None and self._cancellation_registry is not None:
            self._cancellation_registry.register(
                claimed_run.run_id,
                execution_task,
            )

        heartbeat_task = asyncio.create_task(
            heartbeat_loop(
                self._agent_run_repository,
                run_id=claimed_run.run_id,
                lease_id=lease_id,
                lease_seconds=self._lease_seconds,
                ownership_lost=ownership_lost,
            )
        )

        try:
            current_run = self._agent_run_repository.get(claimed_run.run_id)

            if current_run is not None and current_run.cancellation_requested:
                execution_task.cancel()

            response = await self._runtime.resume(
                claimed_run.agent_name,
                request,
                checkpoint,
                run_id=claimed_run.run_id,
                lease_id=lease_id,
                execution_ownership_lost=ownership_lost,
            )

        except asyncio.CancelledError:
            cancelled_at = datetime.now(UTC)

            self._agent_run_repository.cancel_if_owner(
                claimed_run.run_id,
                lease_id=lease_id,
                completed_at=cancelled_at,
            )

            raise

        except AgentExecutionOwnershipLostError:
            raise

        except AgentExecutionControlSignal:
            paused_at = datetime.now(UTC)

            paused_run = self._agent_run_repository.transition_to_waiting_for_approval_if_owner(
                claimed_run.run_id,
                lease_id=lease_id,
                updated_at=paused_at,
            )

            if paused_run is None:
                raise RuntimeError(
                    f"Agent run '{claimed_run.run_id}' lost lease ownership "
                    "while pausing for approval."
                )

            raise

        except Exception as exc:
            self._agent_run_repository.fail_if_owner(
                claimed_run.run_id,
                lease_id=lease_id,
                completed_at=datetime.now(UTC),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

        finally:
            heartbeat_task.cancel()

            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            if execution_task is not None and self._cancellation_registry is not None:
                self._cancellation_registry.unregister(
                    claimed_run.run_id,
                    execution_task,
                )

        completed_run = self._agent_run_repository.complete_if_owner(
            claimed_run.run_id,
            lease_id=lease_id,
            completed_at=datetime.now(UTC),
            output=response.output,
        )

        if completed_run is None:
            raise AgentExecutionOwnershipLostError(
                f"Agent run '{claimed_run.run_id}' lost lease ownership "
                "before approval continuation completion."
            )

        return response

    @staticmethod
    def _validate_checkpoint(
        checkpoint: AgentExecutionCheckpoint,
        *,
        approval_run_id: str,
    ) -> None:
        if checkpoint.run_id != approval_run_id:
            raise ValueError(
                "Approval continuation checkpoint run_id does not match "
                "the approval request run_id."
            )

        if checkpoint.position is not AgentCheckpointPosition.BEFORE_TOOL_EXECUTION:
            raise ValueError("Approval continuation requires a BEFORE_TOOL_EXECUTION checkpoint.")
