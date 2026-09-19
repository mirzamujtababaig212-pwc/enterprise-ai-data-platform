from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ai_platform.agents.checkpoint import (
    AgentCheckpointHandler,
    AgentExecutionCheckpoint,
)
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)

logger = logging.getLogger(__name__)


class PostgreSQLAgentCheckpointHandler(AgentCheckpointHandler):
    """Best-effort durable checkpoint handler for agent execution state."""

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
    ) -> None:
        session = self._session_factory()

        try:
            repository = PostgreSQLAgentCheckpointsRepository(session)
            repository.save(checkpoint)
        except Exception:
            logger.exception(
                "Failed to persist agent execution checkpoint: run_id=%s "
                "agent_name=%s tool_round=%s position=%s",
                checkpoint.run_id,
                checkpoint.agent_name,
                checkpoint.tool_round,
                checkpoint.position.value,
            )
        finally:
            session.close()
