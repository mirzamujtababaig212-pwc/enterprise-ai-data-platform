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


def assistant_tool_calls_from_message(
    message: AgentMessage,
) -> tuple[AgentToolCall, ...]:
    """
    Decode provider-neutral tool calls from a canonical assistant message.

    Checkpoint recovery uses this representation so tool calls can be
    reconstructed without depending on an LLM provider's native message
    format.
    """
    if not isinstance(message, AgentMessage):
        raise TypeError("Assistant tool-call message must be an AgentMessage.")

    if message.role is not AgentMessageRole.ASSISTANT:
        raise ValueError("Assistant tool-call message must have assistant role.")

    try:
        payload = json.loads(message.content)
    except json.JSONDecodeError as exc:
        raise ValueError("Assistant tool-call message content must contain valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Assistant tool-call message payload must be a dictionary.")

    raw_tool_calls = payload.get("tool_calls")

    if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
        raise ValueError("Assistant tool-call message tool_calls must be a non-empty list.")

    tool_calls: list[AgentToolCall] = []

    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            raise ValueError("Assistant tool-call message tool_calls must contain dictionaries.")

        call_id = raw_tool_call.get("call_id")
        name = raw_tool_call.get("name")
        arguments = raw_tool_call.get("arguments", {})

        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Assistant tool-call message call_id must be a non-empty string.")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Assistant tool-call message name must be a non-empty string.")

        if not isinstance(arguments, dict):
            raise ValueError("Assistant tool-call message arguments must be a dictionary.")

        tool_calls.append(
            AgentToolCall(
                call_id=call_id,
                name=name,
                arguments=dict(arguments),
            )
        )

    return tuple(tool_calls)


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
