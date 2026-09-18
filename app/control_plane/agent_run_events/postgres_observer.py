from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ai_platform.agents.observability import AgentExecutionEvent
from ai_platform.agents.observer import AgentExecutionObserver
from app.control_plane.agent_run_events.postgres_repository import (
    PostgreSQLAgentRunEventsRepository,
)

logger = logging.getLogger(__name__)


class PostgreSQLAgentRunEventObserver(AgentExecutionObserver):
    """Best-effort durable observer for agent execution history."""

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def record(self, event: AgentExecutionEvent) -> None:
        if event.run_id is None:
            return

        session = self._session_factory()

        try:
            repository = PostgreSQLAgentRunEventsRepository(session)
            repository.record(event)
        except Exception:
            logger.exception(
                "Failed to persist agent execution event: run_id=%s event_type=%s",
                event.run_id,
                event.event_type.value,
            )
        finally:
            session.close()
