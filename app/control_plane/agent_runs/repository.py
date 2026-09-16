from __future__ import annotations

from typing import Protocol

from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus


class AgentRunRepository(Protocol):
    def create(
        self,
        run: AgentRun,
        *,
        commit: bool = True,
    ) -> AgentRun: ...

    def get(self, run_id: str) -> AgentRun | None: ...

    def update(
        self,
        run: AgentRun,
        *,
        commit: bool = True,
    ) -> AgentRun: ...

    def list(
        self,
        *,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]: ...
