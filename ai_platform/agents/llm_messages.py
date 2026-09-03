from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Any
from ai_platform.agents.tool_calls import AgentToolCall


class AgentMessageRole(StrEnum):
    """
    Roles supported by the agent-side LLM conversation contract.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class AgentMessage:
    """
    Immutable message used by the Agent layer when constructing
    an LLM conversation.
    """

    role: AgentMessageRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Agent message content must not be empty.")


def system_message(content: str) -> AgentMessage:
    """
    Construct a system message.
    """
    return AgentMessage(
        role=AgentMessageRole.SYSTEM,
        content=content,
    )


def user_message(content: str) -> AgentMessage:
    """
    Construct a user message.
    """
    return AgentMessage(
        role=AgentMessageRole.USER,
        content=content,
    )


def assistant_message(content: str) -> AgentMessage:
    """
    Construct an assistant message.
    """
    return AgentMessage(
        role=AgentMessageRole.ASSISTANT,
        content=content,
    )


def tool_message(content: str) -> AgentMessage:
    """
    Construct a tool-result message.
    """
    return AgentMessage(
        role=AgentMessageRole.TOOL,
        content=content,
    )


def tool_result_message(
    *,
    call_id: str,
    tool_name: str,
    output: object = None,
    error: str | None = None,
) -> AgentMessage:
    """
    Construct a canonical tool-result message.

    The message remains provider-neutral while preserving the
    identity and outcome of the originating tool call.
    """

    if not call_id.strip():
        raise ValueError("Tool result message call_id must not be empty.")

    if not tool_name.strip():
        raise ValueError("Tool result message tool_name must not be empty.")

    if error is not None and not error.strip():
        raise ValueError("Tool result message error must not be empty.")

    payload = {
        "call_id": call_id,
        "tool_name": tool_name,
        "success": error is None,
    }

    if error is not None:
        payload["error"] = error
    else:
        payload["output"] = output

    return tool_message(
        json.dumps(
            payload,
            default=str,
            sort_keys=True,
        )
    )


def assistant_tool_call_message(
    *,
    tool_calls: tuple[AgentToolCall, ...],
    content: str = "",
) -> AgentMessage:
    """
    Construct an assistant message representing one or more
    provider-neutral tool calls.

    The tool-call metadata is serialized into the message content
    temporarily so the AgentMessage contract remains immutable and
    provider-neutral.
    """

    if not tool_calls:
        raise ValueError("Assistant tool-call message must contain at least one tool call.")

    for tool_call in tool_calls:
        if not isinstance(tool_call, AgentToolCall):
            raise TypeError("Assistant tool-call message must contain " "AgentToolCall instances.")

    payload: dict[str, Any] = {
        "tool_calls": [
            {
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": dict(tool_call.arguments),
            }
            for tool_call in tool_calls
        ],
    }

    if content.strip():
        payload["content"] = content

    return AgentMessage(
        role=AgentMessageRole.ASSISTANT,
        content=json.dumps(
            payload,
            default=str,
            sort_keys=True,
        ),
    )
