from __future__ import annotations

from datetime import datetime

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

    def claim_for_recovery(
        self,
        run_id: str,
        *,
        started_at: datetime,
        lease_id: str,
        lease_expires_at: datetime,
    ) -> AgentRun | None: ...

    def heartbeat(
        self,
        run_id: str,
        *,
        lease_id: str,
        lease_expires_at: datetime,
    ) -> AgentRun | None: ...

    def claim_expired_running_run(
        self,
        run_id: str,
        *,
        stale_before: datetime,
        started_at: datetime,
        lease_id: str,
        lease_expires_at: datetime,
    ) -> AgentRun | None: ...

    def list(
        self,
        *,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]: ...
