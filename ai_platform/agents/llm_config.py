from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentLLMConfig:
    """
    LLM configuration bound to an AgentDefinition.

    This contains agent-level configuration only.

    Provider routing, fallback, retries, authentication,
    observability, and provider execution remain owned by
    the LLM Gateway.
    """

    model: str | None = None
    system_prompt: str = ""

    temperature: float = 0.7
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.model is not None and not self.model.strip():
            raise ValueError("Agent LLM model must not be empty when provided.")

        if not self.system_prompt.strip():
            raise ValueError("Agent LLM system prompt must not be empty.")

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Agent LLM temperature must be between 0 and 2.")

        if self.max_tokens <= 0:
            raise ValueError("Agent LLM max_tokens must be greater than zero.")
