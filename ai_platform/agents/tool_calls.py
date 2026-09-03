from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentToolCall:
    """
    Provider-neutral representation of an LLM-requested tool call.

    The Agent layer must not depend on provider-specific response objects.
    Providers are responsible for translating their native tool-call
    representation into this contract.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("Agent tool call_id must not be empty.")

        if not self.name.strip():
            raise ValueError("Agent tool name must not be empty.")

        if self.arguments is None:
            raise ValueError("Agent tool call arguments must not be None.")

        if not isinstance(self.arguments, dict):
            raise TypeError("Agent tool call arguments must be a dictionary.")

        object.__setattr__(
            self,
            "arguments",
            dict(self.arguments),
        )


@dataclass(frozen=True)
class AgentToolResult:
    """
    Provider-neutral result of executing an LLM-requested tool call.

    call_id correlates the result with the originating AgentToolCall.
    """

    call_id: str
    tool_name: str
    output: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("Agent tool result call_id must not be empty.")

        if not self.tool_name.strip():
            raise ValueError("Agent tool result tool_name must not be empty.")

        if self.error is not None and not isinstance(
            self.error,
            str,
        ):
            raise TypeError("Agent tool result error must be a string or None.")

        if self.error is not None and not self.error.strip():
            raise ValueError("Agent tool result error must not be empty.")

    @property
    def success(self) -> bool:
        return self.error is None
