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

    def get_for_tenant(
        self,
        run_id: str,
        tenant_id: str,
    ) -> AgentRun | None: ...

    def get_by_idempotency_key(
        self,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
    ) -> AgentRun | None: ...

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
        max_recovery_attempts: int,
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
        max_recovery_attempts: int,
    ) -> AgentRun | None: ...

    def fail_recovery_exhausted(
        self,
        run_id: str,
        *,
        completed_at: datetime,
        max_recovery_attempts: int,
        error_type: str,
        error_message: str,
    ) -> AgentRun | None: ...

    def list_expired_running_runs(
        self,
        *,
        stale_before: datetime,
        limit: int = 100,
    ) -> list[AgentRun]: ...

    def complete_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
        output,
    ) -> AgentRun | None: ...

    def fail_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
        error_type: str,
        error_message: str,
    ) -> AgentRun | None: ...

    def cancel_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
    ) -> AgentRun | None: ...

    def list(
        self,
        *,
        tenant_id: str | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]: ...

    def request_cancellation(
        self,
        run_id: str,
        *,
        requested_at: datetime,
    ) -> AgentRun | None: ...
