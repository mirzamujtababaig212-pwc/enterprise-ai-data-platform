from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from ai_platform.agents.exceptions import AgentExecutionOwnershipLostError
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from ai_platform.agents.runtime import AgentRuntime

from app.control_plane.agent_checkpoints.repository import (
    AgentCheckpointsRepository,
)
from app.control_plane.agent_runs.exceptions import RecoveryExhaustedError
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.cancellation import AgentRunCancellationRegistry
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.lease import (
    create_lease,
    heartbeat_loop,
)


@dataclass(frozen=True)
class AgentRunRecoverySweepResult:
    """Outcome of one automatic stale-run recovery sweep."""

    recovered: tuple[AgentRunExecutionResult, ...]
    failed_run_ids: tuple[str, ...]


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
        observer: AgentExecutionObserver | None = None,
        cancellation_registry: AgentRunCancellationRegistry | None = None,
        lease_seconds: int = 60,
        max_recovery_attempts: int = 3,
    ) -> None:
        if max_recovery_attempts <= 0:
            raise ValueError("max_recovery_attempts must be greater than zero.")
        self._runtime = runtime
        self._cancellation_registry = cancellation_registry
        self._repository = repository
        self._checkpoints_repository = checkpoints_repository
        self._observer = observer
        self._lease_seconds = lease_seconds
        self._max_recovery_attempts = max_recovery_attempts

    async def _emit(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        if self._observer is None:
            return

        try:
            await self._observer.record(event)
        except Exception:
            # Recovery observability must never change recovery semantics.
            pass

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

        if existing_run.recovery_attempts >= self._max_recovery_attempts:
            raise RecoveryExhaustedError(
                f"Agent run '{run_id}' exhausted the maximum of "
                f"{self._max_recovery_attempts} recovery attempts."
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
            max_recovery_attempts=self._max_recovery_attempts,
        )

        if run is None:
            raise RuntimeError(
                f"Agent run '{run_id}' could not be claimed for recovery.",
            )

        await self._emit(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_RECOVERY_STARTED,
                agent_name=run.agent_name,
                run_id=run.run_id,
                session_id=run.session_id,
                user_id=run.user_id,
                metadata={
                    "recovery_type": "failed_run",
                },
            )
        )

        return await self._resume_claimed_run(
            run,
            recovery_type="failed_run",
        )

    async def recover_stale_runs(
        self,
        *,
        stale_before: datetime | None = None,
        limit: int = 100,
    ) -> AgentRunRecoverySweepResult:
        if limit <= 0:
            raise ValueError("Recovery limit must be greater than zero.")

        recovery_cutoff = stale_before or datetime.now(UTC)

        candidates = self._repository.list_expired_running_runs(
            stale_before=recovery_cutoff,
            limit=limit,
        )

        results: list[AgentRunExecutionResult] = []
        failed_run_ids: list[str] = []

        for candidate in candidates:
            claimed_at = datetime.now(UTC)
            lease_id, lease_expires_at = create_lease(self._lease_seconds)

            if candidate.recovery_attempts >= self._max_recovery_attempts:
                exhausted_at = datetime.now(UTC)
                exhausted_run = self._repository.fail_recovery_exhausted(
                    candidate.run_id,
                    completed_at=exhausted_at,
                    max_recovery_attempts=self._max_recovery_attempts,
                    error_type=RecoveryExhaustedError.__name__,
                    error_message=(
                        f"Agent run '{candidate.run_id}' exhausted "
                        f"the maximum of {self._max_recovery_attempts} recovery attempts."
                    ),
                )

                if exhausted_run is not None:
                    await self._emit(
                        AgentExecutionEvent(
                            event_type=AgentExecutionEventType.AGENT_RECOVERY_FAILED,
                            agent_name=exhausted_run.agent_name,
                            run_id=exhausted_run.run_id,
                            session_id=exhausted_run.session_id,
                            user_id=exhausted_run.user_id,
                            metadata={
                                "recovery_type": "stale_run",
                                "error_type": RecoveryExhaustedError.__name__,
                                "reason": "max_recovery_attempts_exceeded",
                            },
                        )
                    )

                continue

            run = self._repository.claim_expired_running_run(
                candidate.run_id,
                stale_before=recovery_cutoff,
                started_at=claimed_at,
                lease_id=lease_id,
                lease_expires_at=lease_expires_at,
                max_recovery_attempts=self._max_recovery_attempts,
            )

            if run is None:
                continue

            await self._emit(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.AGENT_RECOVERY_STARTED,
                    agent_name=run.agent_name,
                    run_id=run.run_id,
                    session_id=run.session_id,
                    user_id=run.user_id,
                    metadata={
                        "recovery_type": "stale_run",
                    },
                )
            )

            try:
                results.append(
                    await self._resume_claimed_run(
                        run,
                        recovery_type="stale_run",
                    )
                )
            except Exception:
                failed_run_ids.append(candidate.run_id)
                continue

        return AgentRunRecoverySweepResult(
            recovered=tuple(results),
            failed_run_ids=tuple(failed_run_ids),
        )

    async def _resume_claimed_run(
        self,
        run: AgentRun,
        *,
        recovery_type: str,
    ) -> AgentRunExecutionResult:
        if run.lease_id is None:
            raise RuntimeError(
                f"Agent run '{run.run_id}' was claimed without a lease.",
            )

        if run.request_snapshot is None:
            exc = RuntimeError(
                f"Agent run '{run.run_id}' has no request snapshot and " "cannot be recovered.",
            )
            await self._mark_failed(run, exc)
            raise exc

        ownership_lost = asyncio.Event()
        execution_task = asyncio.current_task()

        if execution_task is not None and self._cancellation_registry is not None:
            self._cancellation_registry.register(run.run_id, execution_task)

        heartbeat_task = asyncio.create_task(
            heartbeat_loop(
                self._repository,
                run_id=run.run_id,
                lease_id=run.lease_id,
                lease_seconds=self._lease_seconds,
                ownership_lost=ownership_lost,
            )
        )

        try:
            if execution_task is not None:
                current_run = self._repository.get(run.run_id)
                if current_run is not None and current_run.cancellation_requested:
                    execution_task.cancel()

            checkpoint = self._checkpoints_repository.get_latest(run.run_id)

            if checkpoint is None:
                raise RuntimeError(
                    f"Agent run '{run.run_id}' has no execution checkpoint "
                    "and cannot be recovered.",
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
                lease_id=run.lease_id,
                execution_ownership_lost=ownership_lost,
            )
        except asyncio.CancelledError:
            cancelled_at = datetime.now(UTC)

            cancelled_run = self._repository.cancel_if_owner(
                run.run_id,
                lease_id=run.lease_id,
                completed_at=cancelled_at,
            )

            if cancelled_run is not None:
                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.AGENT_CANCELLED,
                        agent_name=run.agent_name,
                        run_id=run.run_id,
                        session_id=run.session_id,
                        user_id=run.user_id,
                        metadata={
                            "recovery_type": recovery_type,
                        },
                    )
                )

            raise
        except AgentExecutionOwnershipLostError:
            raise
        except Exception as exc:
            exhausted = run.recovery_attempts >= self._max_recovery_attempts
            failure = (
                RecoveryExhaustedError(
                    f"Agent run '{run.run_id}' exhausted the maximum of "
                    f"{self._max_recovery_attempts} recovery attempts: {exc}"
                )
                if exhausted
                else exc
            )

            await self._mark_failed(run, failure)
            await self._emit(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.AGENT_RECOVERY_FAILED,
                    agent_name=run.agent_name,
                    run_id=run.run_id,
                    session_id=run.session_id,
                    user_id=run.user_id,
                    metadata={
                        "recovery_type": recovery_type,
                        "error_type": type(failure).__name__,
                        **({"reason": "max_recovery_attempts_exceeded"} if exhausted else {}),
                    },
                )
            )

            if exhausted:
                raise failure from exc

            raise
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            if execution_task is not None and self._cancellation_registry is not None:
                self._cancellation_registry.unregister(
                    run.run_id,
                    execution_task,
                )

        completed_at = datetime.now(UTC)

        completed_run = self._repository.complete_if_owner(
            run.run_id,
            lease_id=run.lease_id,
            completed_at=completed_at,
            output=response.output,
        )

        if completed_run is None:
            raise AgentExecutionOwnershipLostError(
                f"Agent run '{run.run_id}' lost lease ownership before completion.",
            )

        await self._emit(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_RECOVERY_COMPLETED,
                agent_name=run.agent_name,
                run_id=run.run_id,
                session_id=run.session_id,
                user_id=run.user_id,
                metadata={
                    "recovery_type": recovery_type,
                },
            )
        )

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
