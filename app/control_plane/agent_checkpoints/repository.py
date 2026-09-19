from __future__ import annotations

from typing import Protocol

from ai_platform.agents.checkpoint import AgentExecutionCheckpoint


class AgentCheckpointsRepository(Protocol):
    def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
        *,
        commit: bool = True,
    ) -> AgentExecutionCheckpoint: ...

    def get_latest(
        self,
        run_id: str,
    ) -> AgentExecutionCheckpoint | None: ...
