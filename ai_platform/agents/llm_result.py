from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from ai_platform.agents.tool_calls import AgentToolCall


@dataclass(frozen=True)
class AgentLLMUsage:
    """Normalized token usage returned by an LLM Gateway."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("LLM prompt_tokens must be greater than or equal to zero.")

        if self.completion_tokens < 0:
            raise ValueError("LLM completion_tokens must be greater than or equal to zero.")

        if self.total_tokens < 0:
            raise ValueError("LLM total_tokens must be greater than or equal to zero.")


@dataclass(frozen=True)
class AgentLLMResult:
    """Provider-neutral result exposed to an Agent."""

    text: str
    provider: str
    model: str
    usage: AgentLLMUsage
    raw_response: dict[str, Any] | None = None
    tool_calls: tuple[AgentToolCall, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Agent LLM result text must be a string.")

        if not self.text.strip() and not self.tool_calls:
            raise ValueError(
                "Agent LLM result text must not be empty " "when no tool calls are present."
            )

        if not self.provider.strip():
            raise ValueError("LLM result provider must not be empty.")

        if not self.model.strip():
            raise ValueError("LLM result model must not be empty.")

        for tool_call in self.tool_calls:
            if not isinstance(tool_call, AgentToolCall):
                raise TypeError("LLM result tool_calls must contain AgentToolCall instances.")

        object.__setattr__(
            self,
            "tool_calls",
            tuple(self.tool_calls),
        )
