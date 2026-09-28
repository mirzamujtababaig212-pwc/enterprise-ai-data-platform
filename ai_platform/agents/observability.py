from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentExecutionEventType(StrEnum):
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    AGENT_RECOVERY_STARTED = "agent.recovery.started"
    AGENT_RECOVERY_COMPLETED = "agent.recovery.completed"
    AGENT_RECOVERY_FAILED = "agent.recovery.failed"
    AGENT_CANCELLED = "agent.cancelled"

    RUNTIME_DECISION = "runtime.decision"

    LLM_REQUESTED = "llm.requested"
    LLM_COMPLETED = "llm.completed"

    TOOL_CALL_REQUESTED = "tool.call.requested"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"

    TOOL_AUTHORIZATION_DECISION = "tool.authorization.decision"
    GOVERNANCE_DECISION = "governance.decision"
    APPROVAL_DECISION = "approval.decision"

    ORCHESTRATION_STEP_STARTED = "orchestration.step.started"
    ORCHESTRATION_STEP_COMPLETED = "orchestration.step.completed"
    ORCHESTRATION_STEP_FAILED = "orchestration.step.failed"
    ORCHESTRATION_STEP_CANCELLED = "orchestration.step.cancelled"

    MEMORY_RETRIEVAL_STARTED = "memory.retrieval.started"
    MEMORY_RETRIEVAL_COMPLETED = "memory.retrieval.completed"
    MEMORY_RETRIEVAL_FAILED = "memory.retrieval.failed"
    MEMORY_WRITE_STARTED = "memory.write.started"
    MEMORY_WRITE_COMPLETED = "memory.write.completed"
    MEMORY_WRITE_FAILED = "memory.write.failed"


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
    run_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    principal: str | None = None
    tool_round: int | None = None
    tool_name: str | None = None
    call_id: str | None = None
    provider: str | None = None
    model: str | None = None
    step_id: str | None = None
    step_index: int | None = None
    step_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.event_type,
            AgentExecutionEventType,
        ):
            raise TypeError("Agent execution event_type must be an " "AgentExecutionEventType.")

        if not self.agent_name.strip():
            raise ValueError("Agent execution event agent_name must not be empty.")

        if self.run_id is not None:
            if not isinstance(self.run_id, str):
                raise TypeError("Agent execution event run_id must be a string or None.")

            if not self.run_id.strip():
                raise ValueError("Agent execution event run_id must not be empty.")

        if self.session_id is not None:
            if not isinstance(self.session_id, str):
                raise TypeError("Agent execution event session_id must be a string or None.")

            if not self.session_id.strip():
                raise ValueError("Agent execution event session_id must not be empty.")

        if self.user_id is not None:
            if not isinstance(self.user_id, str):
                raise TypeError("Agent execution event user_id must be a string or None.")

            if not self.user_id.strip():
                raise ValueError("Agent execution event user_id must not be empty.")

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

        if self.step_id is not None:
            if not isinstance(self.step_id, str):
                raise TypeError("Agent execution event step_id must be a string or None.")

            if not self.step_id.strip():
                raise ValueError("Agent execution event step_id must not be empty.")

        if self.step_index is not None:
            if not isinstance(self.step_index, int):
                raise TypeError("Agent execution event step_index must be an integer or None.")

            if self.step_index < 0:
                raise ValueError("Agent execution event step_index must be >= 0.")

        if self.step_name is not None:
            if not isinstance(self.step_name, str):
                raise TypeError("Agent execution event step_name must be a string or None.")

            if not self.step_name.strip():
                raise ValueError("Agent execution event step_name must not be empty.")

        if not isinstance(self.metadata, dict):
            raise TypeError("Agent execution event metadata must be a dictionary.")

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )
