from __future__ import annotations

from dataclasses import dataclass

from ai_platform.agents.observability import AgentExecutionEvent


@dataclass
class AgentRunEventsPage:
    events: list[AgentExecutionEvent]
    next_cursor: str | None
    has_more: bool
