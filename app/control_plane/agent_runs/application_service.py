from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from ai_platform.agents.exceptions import AgentExecutionOwnershipLostError
from ai_platform.agents.models import AgentRequest, AgentResponse
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from ai_platform.agents.runtime import AgentRuntime
from app.control_plane.agent_runs.cancellation import (
    AgentRunCancellationRegistry,
)
from app.control_plane.agent_runs.admission import (
    AgentRunAdmissionPolicy,
    AllowAllAgentRunAdmissionPolicy,
)
from app.control_plane.agent_runs.exceptions import (
    AgentRunAdmissionRejectedError,
    AgentRunIdempotencyConflictError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.agent_run_events.repository import (
    AgentRunEventsRepository,
)
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.lease import (
    create_lease,
    heartbeat_loop,
)


class AgentRunApplicationService:
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        repository: AgentRunRepository,
        events_repository: AgentRunEventsRepository | None = None,
        observer: AgentExecutionObserver | None = None,
        admission_policy: AgentRunAdmissionPolicy | None = None,
        cancellation_registry: AgentRunCancellationRegistry | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self._runtime = runtime
        self._repository = repository
        self._events_repository = events_repository
        self._observer = observer
        self._admission_policy = (
            admission_policy if admission_policy is not None else AllowAllAgentRunAdmissionPolicy()
        )
        self._lease_seconds = lease_seconds
        self._cancellation_registry = cancellation_registry

    async def _emit(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        if self._observer is None:
            return

        try:
            await self._observer.record(event)
        except Exception:
            # Cancellation observability must never change cancellation semantics.
            pass

    @staticmethod
    def _build_replayed_response(
        run: AgentRun,
    ) -> AgentResponse:
        return AgentResponse(
            agent_name=run.agent_name,
            output=run.output,
            session_id=run.session_id,
            metadata=dict(run.metadata),
        )

    @staticmethod
    def _request_snapshots_match(
        existing_run: AgentRun,
        request: AgentRequest,
    ) -> bool:
        if existing_run.request_snapshot is None:
            return False

        if existing_run.session_id != request.session_id:
            return False

        return existing_run.request_snapshot == AgentRunRequestSnapshot.from_request(request)

    def _resolve_existing_idempotent_run(
        self,
        *,
        existing_run: AgentRun,
        agent_name: str,
        request: AgentRequest,
    ) -> AgentRunExecutionResult:
        if existing_run.agent_name != agent_name:
            raise AgentRunIdempotencyConflictError(
                "Idempotency key is already associated with a different agent run."
            )

        if not self._request_snapshots_match(existing_run, request):
            raise AgentRunIdempotencyConflictError(
                "Idempotency key is already associated with a different request."
            )

        if existing_run.status is AgentRunStatus.FAILED:
            message = existing_run.error_message or "Agent run failed."
            raise RuntimeError(
                f"Idempotent agent run '{existing_run.run_id}' previously failed: {message}"
            )

        return AgentRunExecutionResult(
            run_id=existing_run.run_id,
            response=self._build_replayed_response(existing_run),
        )

    async def execute(
        self,
        *,
        agent_name: str,
        request: AgentRequest,
        idempotency_key: str | None = None,
    ) -> AgentRunExecutionResult:
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()

            if not idempotency_key:
                raise ValueError("Idempotency key must not be empty when provided.")

            if request.user_id is None:
                raise ValueError("A user_id is required when an idempotency key is provided.")

            existing_run = self._repository.get_by_idempotency_key(
                request.user_id,
                idempotency_key,
            )

            if existing_run is not None:
                return self._resolve_existing_idempotent_run(
                    existing_run=existing_run,
                    agent_name=agent_name,
                    request=request,
                )

        run = AgentRun(
            run_id=str(uuid4()),
            agent_name=agent_name,
            session_id=request.session_id,
            user_id=request.user_id,
            principal=request.principal,
            idempotency_key=idempotency_key,
            status=AgentRunStatus.PENDING,
            metadata=dict(request.metadata),
            request_snapshot=AgentRunRequestSnapshot.from_request(request),
        )

        try:
            self._repository.create(run)
        except DuplicateAgentRunError:
            if idempotency_key is None or request.user_id is None:
                raise

            existing_run = self._repository.get_by_idempotency_key(
                request.user_id,
                idempotency_key,
            )

            if existing_run is None:
                raise RuntimeError(
                    "Agent run creation conflicted, but the idempotent run " "could not be loaded."
                )

            return self._resolve_existing_idempotent_run(
                existing_run=existing_run,
                agent_name=agent_name,
                request=request,
            )

        admission = await self._admission_policy.evaluate(
            agent_name=agent_name,
            request=request,
            run=run,
        )

        if not admission.allowed:
            rejected_at = datetime.now(UTC)
            rejected_run = run.transition_to(AgentRunStatus.REJECTED).model_copy(
                update={
                    "completed_at": rejected_at,
                    "metadata": {
                        **run.metadata,
                        "admission": {
                            "allowed": False,
                            "reason": admission.reason,
                        },
                    },
                }
            )

            self._repository.update(rejected_run)

            reason = admission.reason or "Agent run admission was rejected."
            raise AgentRunAdmissionRejectedError(reason)

        started_at = datetime.now(UTC)
        lease_id, lease_expires_at = create_lease(self._lease_seconds)

        run = run.transition_to(AgentRunStatus.RUNNING).model_copy(
            update={
                "started_at": started_at,
                "lease_id": lease_id,
                "lease_expires_at": lease_expires_at,
            }
        )

        self._repository.update(run)

        ownership_lost = asyncio.Event()

        heartbeat_task = asyncio.create_task(
            heartbeat_loop(
                self._repository,
                run_id=run.run_id,
                lease_id=lease_id,
                lease_seconds=self._lease_seconds,
                ownership_lost=ownership_lost,
            )
        )

        execution_task = asyncio.current_task()
        if execution_task is None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            raise RuntimeError("Agent run execution is not attached to an asyncio task.")

        if self._cancellation_registry is not None:
            self._cancellation_registry.register(run.run_id, execution_task)

        try:
            current_run = self._repository.get(run.run_id)
            if current_run is not None and current_run.cancellation_requested:
                execution_task.cancel()

            response = await self._runtime.run(
                agent_name,
                request,
                lease_id=lease_id,
                run_id=run.run_id,
                execution_ownership_lost=ownership_lost,
            )
        except asyncio.CancelledError:
            cancelled_at = datetime.now(UTC)

            try:
                cancelled_run = self._repository.cancel_if_owner(
                    run.run_id,
                    lease_id=lease_id,
                    completed_at=cancelled_at,
                )
            except Exception:
                # Cancellation must continue to propagate even if terminal
                # cancellation persistence fails.
                pass
            else:
                if cancelled_run is not None:
                    await self._emit(
                        AgentExecutionEvent(
                            event_type=AgentExecutionEventType.AGENT_CANCELLED,
                            agent_name=run.agent_name,
                            run_id=run.run_id,
                            session_id=run.session_id,
                            user_id=run.user_id,
                        )
                    )

            raise
        except AgentExecutionOwnershipLostError:
            raise
        except Exception as exc:
            failed_at = datetime.now(UTC)

            try:
                persisted_failed_run = self._repository.fail_if_owner(
                    run.run_id,
                    lease_id=lease_id,
                    completed_at=failed_at,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception:
                # Preserve the original execution failure if terminal
                # failure persistence itself fails.
                pass
            else:
                if persisted_failed_run is None:
                    raise RuntimeError(
                        f"Agent run '{run.run_id}' lost lease ownership while failing."
                    ) from exc

            raise
        finally:
            if self._cancellation_registry is not None:
                self._cancellation_registry.unregister(
                    run.run_id,
                    execution_task,
                )

            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        completed_at = datetime.now(UTC)
        completed_run = self._repository.complete_if_owner(
            run.run_id,
            lease_id=lease_id,
            completed_at=completed_at,
            output=response.output,
        )

        if completed_run is None:
            raise RuntimeError(f"Agent run '{run.run_id}' lost lease ownership before completion.")

        return AgentRunExecutionResult(
            run_id=completed_run.run_id,
            response=response,
        )

    def get_run(self, run_id: str) -> AgentRun | None:
        return self._repository.get(run_id)

    def cancel(self, run_id: str) -> AgentRun:
        run = self._repository.get(run_id)

        if run is None:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        if run.status is not AgentRunStatus.RUNNING:
            raise ValueError(
                f"Agent run '{run_id}' is not cancellable from status " f"'{run.status.value}'.",
            )

        requested = self._repository.request_cancellation(
            run_id,
            requested_at=datetime.now(UTC),
        )

        if requested is None:
            raise RuntimeError(
                f"Agent run '{run_id}' could not accept cancellation.",
            )

        if self._cancellation_registry is not None:
            self._cancellation_registry.cancel(run_id)

        return requested

    def list_runs(
        self,
        *,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]:
        return self._repository.list(
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            status=status,
            limit=limit,
        )

    def list_events(
        self,
        run_id: str,
        *,
        limit: int = 100,
    ) -> list[AgentExecutionEvent]:
        run = self._repository.get(run_id)

        if run is None:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        if self._events_repository is None:
            raise RuntimeError(
                "Agent run events repository is not configured.",
            )

        return self._events_repository.list(
            run_id,
            limit=limit,
        )
