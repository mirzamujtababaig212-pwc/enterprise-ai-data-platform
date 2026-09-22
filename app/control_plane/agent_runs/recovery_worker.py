from __future__ import annotations

from app.control_plane.agent_runs.recovery_service import (
    AgentRunRecoveryService,
    AgentRunRecoverySweepResult,
)


class AgentRunRecoveryWorker:
    """
    Orchestration boundary for automatic stale agent-run recovery.

    The worker owns sweep configuration and invocation only. Recovery
    semantics, lease ownership, checkpoint resume, heartbeat, and terminal
    persistence remain owned by AgentRunRecoveryService.
    """

    def __init__(
        self,
        *,
        recovery_service: AgentRunRecoveryService,
        limit: int = 100,
    ) -> None:
        if limit <= 0:
            raise ValueError("Recovery worker limit must be greater than zero.")

        self._recovery_service = recovery_service
        self._limit = limit

    async def run_once(self) -> AgentRunRecoverySweepResult:
        return await self._recovery_service.recover_stale_runs(
            limit=self._limit,
        )
