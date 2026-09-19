from __future__ import annotations

from datetime import UTC, datetime

from ai_platform.agents.runtime import AgentRuntime

from app.control_plane.agent_checkpoints.repository import (
    AgentCheckpointsRepository,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.lease import create_lease


class AgentRunRecoveryService:
    """
    Durable recovery boundary for interrupted agent runs.

    This service owns the control-plane recovery lifecycle:

        FAILED
          -> atomically claim as RUNNING
          -> load request snapshot
          -> load latest checkpoint
          -> reconstruct AgentRequest
          -> resume through AgentRuntime
          -> COMPLETED or FAILED

    Runtime dependencies are deliberately reconstructed by AgentRuntime.
    The checkpoint remains the authoritative continuation conversation.
    """

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        repository: AgentRunRepository,
        checkpoints_repository: AgentCheckpointsRepository,
        lease_seconds: int = 60,
    ) -> None:
        self._runtime = runtime
        self._repository = repository
        self._checkpoints_repository = checkpoints_repository
        self._lease_seconds = lease_seconds

    async def recover(
        self,
        run_id: str,
    ) -> AgentRunExecutionResult:
        if not run_id.strip():
            raise ValueError("Agent run_id must not be empty.")

        existing_run = self._repository.get(run_id)

        if existing_run is None:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        if existing_run.status is not AgentRunStatus.FAILED:
            raise ValueError(
                f"Agent run '{run_id}' is not eligible for recovery "
                f"from status '{existing_run.status.value}'.",
            )

        if existing_run.request_snapshot is None:
            raise RuntimeError(
                f"Agent run '{run_id}' has no request snapshot and " "cannot be recovered.",
            )

        claimed_at = datetime.now(UTC)
        lease_id, lease_expires_at = create_lease(self._lease_seconds)

        run = self._repository.claim_for_recovery(
            run_id,
            started_at=claimed_at,
            lease_id=lease_id,
            lease_expires_at=lease_expires_at,
        )

        if run is None:
            raise RuntimeError(
                f"Agent run '{run_id}' could not be claimed for recovery.",
            )

        if run.lease_id is None:
            raise RuntimeError(f"Agent run '{run.run_id}' was claimed without a lease.")

        try:
            checkpoint = self._checkpoints_repository.get_latest(run_id)

            if checkpoint is None:
                raise RuntimeError(
                    f"Agent run '{run_id}' has no execution checkpoint " "and cannot be recovered.",
                )

            request = run.request_snapshot.to_request(
                session_id=run.session_id,
                user_id=run.user_id,
            )

            response = await self._runtime.resume(
                run.agent_name,
                request,
                checkpoint,
                run_id=run.run_id,
            )
        except Exception as exc:
            await self._mark_failed(run, exc)
            raise

        completed_at = datetime.now(UTC)

        completed_run = self._repository.complete_if_owner(
            run.run_id,
            lease_id=run.lease_id,
            completed_at=completed_at,
            output=response.output,
        )

        if completed_run is None:
            raise RuntimeError(f"Agent run '{run.run_id}' lost lease ownership before completion.")

        return AgentRunExecutionResult(
            run_id=completed_run.run_id,
            response=response,
        )

    async def _mark_failed(
        self,
        run: AgentRun,
        exc: Exception,
    ) -> None:
        failed_at = datetime.now(UTC)

        try:
            self._repository.fail_if_owner(
                run.run_id,
                lease_id=run.lease_id,
                completed_at=failed_at,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception:
            # Recovery must preserve the original execution failure.
            pass
