from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ai_platform.agents.models import AgentRequest
from ai_platform.agents.runtime import AgentRuntime

from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.repository import AgentRunRepository


class AgentRunApplicationService:
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        repository: AgentRunRepository,
    ) -> None:
        self._runtime = runtime
        self._repository = repository

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
        )

        self._repository.create(run)

        started_at = datetime.now(UTC)
        run = run.transition_to(AgentRunStatus.RUNNING).model_copy(
            update={
                "started_at": started_at,
            }
        )
        self._repository.update(run)

        try:
            response = await self._runtime.run(
                agent_name,
                request,
                run_id=run.run_id,
            )
        except Exception as exc:
            failed_at = datetime.now(UTC)
            failed_run = run.transition_to(AgentRunStatus.FAILED).model_copy(
                update={
                    "completed_at": failed_at,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            self._repository.update(failed_run)
            raise

        completed_at = datetime.now(UTC)
        completed_run = run.transition_to(AgentRunStatus.COMPLETED).model_copy(
            update={
                "completed_at": completed_at,
                "output": response.output,
            }
        )
        self._repository.update(completed_run)

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
