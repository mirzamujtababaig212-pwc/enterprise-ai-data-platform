from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_platform.agents.checkpoint import (
    AgentCheckpointHandler,
    AgentExecutionCheckpoint,
)
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)


class PostgreSQLAgentCheckpointHandler(AgentCheckpointHandler):
    """Durable PostgreSQL checkpoint handler for agent execution state."""

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
        *,
        lease_id: str | None = None,
    ) -> None:
        session = self._session_factory()

        try:
            repository = PostgreSQLAgentCheckpointsRepository(session)
            repository.save(
                checkpoint,
                lease_id=lease_id,
            )
        finally:
            session.close()
