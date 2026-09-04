from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_platform.agents.llm_config import AgentLLMConfig


@dataclass(frozen=True)
class AgentDefinition:
    """
    Immutable definition of an enterprise AI agent.

    AgentDefinition describes what an agent is and what capabilities
    it is allowed to request. It does not contain runtime state.
    """

    name: str
    description: str
    system_prompt: str

    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024

    tool_names: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name must not be empty.")

        if not self.description.strip():
            raise ValueError("Agent description must not be empty.")

        if not self.system_prompt.strip():
            raise ValueError("Agent system prompt must not be empty.")

        if self.model is not None and not self.model.strip():
            raise ValueError("Agent model must not be empty when provided.")

        if any(not tool_name.strip() for tool_name in self.tool_names):
            raise ValueError("Agent tool names must not contain empty values.")

        if len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("Agent tool names must not contain duplicates.")

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Agent temperature must be between 0 and 2.")

        if self.max_tokens <= 0:
            raise ValueError("Agent max_tokens must be greater than zero.")

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def llm_config(self) -> AgentLLMConfig:
        """
        Return the LLM configuration declared by this agent.

        The configuration is derived from the immutable agent definition.
        """
        return AgentLLMConfig(
            model=self.model,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True)
class AgentRequest:
    """
    Input supplied to an Agent Runtime invocation.
    """

    input: str

    session_id: str | None = None
    user_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.input.strip():
            raise ValueError("Agent request input must not be empty.")

        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("Agent request session_id must not be empty when provided.")

        if self.user_id is not None and not self.user_id.strip():
            raise ValueError("Agent request user_id must not be empty when provided.")

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )


@dataclass(frozen=True)
class AgentResponse:
    """
    Result returned by an Agent Runtime invocation.

    The response model is intentionally transport- and provider-neutral.
    """

    agent_name: str
    output: Any

    session_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_name.strip():
            raise ValueError("Agent response agent_name must not be empty.")

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )
