from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentExecutionEventType(StrEnum):
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    LLM_REQUESTED = "llm.requested"
    LLM_COMPLETED = "llm.completed"

    TOOL_CALL_REQUESTED = "tool.call.requested"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"


@dataclass(frozen=True)
class AgentExecutionEvent:
    """
    Provider-neutral event describing an agent execution lifecycle step.

    Events intentionally contain execution metadata only. Tool arguments
    and tool outputs/results must not be stored here by default because
    they may contain sensitive or high-volume data.
    """

    event_type: AgentExecutionEventType
    agent_name: str
    session_id: str | None = None
    tool_round: int | None = None
    tool_name: str | None = None
    call_id: str | None = None
    provider: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.event_type,
            AgentExecutionEventType,
        ):
            raise TypeError("Agent execution event_type must be an " "AgentExecutionEventType.")

        if not self.agent_name.strip():
            raise ValueError("Agent execution event agent_name must not be empty.")

        if self.session_id is not None:
            if not isinstance(self.session_id, str):
                raise TypeError("Agent execution event session_id must be a string or None.")

            if not self.session_id.strip():
                raise ValueError("Agent execution event session_id must not be empty.")

        if self.tool_round is not None:
            if not isinstance(self.tool_round, int):
                raise TypeError("Agent execution event tool_round must be an integer or None.")

            if self.tool_round < 0:
                raise ValueError("Agent execution event tool_round must be >= 0.")

        if self.tool_name is not None:
            if not isinstance(self.tool_name, str):
                raise TypeError("Agent execution event tool_name must be a string or None.")

            if not self.tool_name.strip():
                raise ValueError("Agent execution event tool_name must not be empty.")

        if self.call_id is not None:
            if not isinstance(self.call_id, str):
                raise TypeError("Agent execution event call_id must be a string or None.")

            if not self.call_id.strip():
                raise ValueError("Agent execution event call_id must not be empty.")

        if self.provider is not None:
            if not isinstance(self.provider, str):
                raise TypeError("Agent execution event provider must be a string or None.")

            if not self.provider.strip():
                raise ValueError("Agent execution event provider must not be empty.")

        if self.model is not None:
            if not isinstance(self.model, str):
                raise TypeError("Agent execution event model must be a string or None.")

            if not self.model.strip():
                raise ValueError("Agent execution event model must not be empty.")

        if not isinstance(self.metadata, dict):
            raise TypeError("Agent execution event metadata must be a dictionary.")

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )
