from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from ai_platform.agents.models import AgentRequest
from ai_platform.agents.observability import AgentExecutionEvent
from ai_platform.agents.runtime import AgentRuntime

from app.control_plane.agent_runs.admission import (
    AgentRunAdmissionPolicy,
    AllowAllAgentRunAdmissionPolicy,
)
from app.control_plane.agent_runs.exceptions import (
    AgentRunAdmissionRejectedError,
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
from app.control_plane.agent_runs.lease import create_lease


class AgentRunApplicationService:
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        repository: AgentRunRepository,
        events_repository: AgentRunEventsRepository | None = None,
        admission_policy: AgentRunAdmissionPolicy | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self._runtime = runtime
        self._repository = repository
        self._events_repository = events_repository
        self._admission_policy = (
            admission_policy if admission_policy is not None else AllowAllAgentRunAdmissionPolicy()
        )
        self._lease_seconds = lease_seconds

    async def _heartbeat_loop(
        self,
        *,
        run_id: str,
        lease_id: str,
    ) -> None:
        interval_seconds = self._lease_seconds / 3

        while True:
            await asyncio.sleep(interval_seconds)

            lease_expires_at = datetime.now(UTC) + timedelta(
                seconds=self._lease_seconds,
            )

            try:
                renewed_run = self._repository.heartbeat(
                    run_id,
                    lease_id=lease_id,
                    lease_expires_at=lease_expires_at,
                )
            except Exception:
                # Heartbeat failures must not mask the agent execution
                # result. Ownership remains enforced by the repository's
                # conditional terminal transition.
                continue

            if renewed_run is None:
                # Another worker owns the run, or the run is no longer
                # RUNNING. Do not attempt to renew it again.
                return

    async def execute(
        self,
        *,
        agent_name: str,
        request: AgentRequest,
    ) -> AgentRunExecutionResult:
        run = AgentRun(
            run_id=str(uuid4()),
            agent_name=agent_name,
            session_id=request.session_id,
            user_id=request.user_id,
            status=AgentRunStatus.PENDING,
            metadata=dict(request.metadata),
            request_snapshot=AgentRunRequestSnapshot.from_request(request),
        )

        self._repository.create(run)

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

        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(
                run_id=run.run_id,
                lease_id=lease_id,
            )
        )

        try:
            response = await self._runtime.run(
                agent_name,
                request,
                run_id=run.run_id,
            )
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
