from __future__ import annotations

import json
from typing import Any, Protocol

from ai_platform.agents.llm_config import AgentLLMConfig
from ai_platform.agents.llm_result import (
    AgentLLMResult,
    AgentLLMUsage,
)
from ai_platform.agents.llm_messages import (
    AgentMessage,
    AgentMessageRole,
    system_message,
    tool_message,
    user_message,
)
from ai_platform.agents.tool_calls import AgentToolCall
from tools.models import ToolDefinition


class LLMGateway(Protocol):
    """
    Minimal agent-facing contract for the existing LLM Gateway.

    The Agent layer depends only on the Gateway's chat operation.
    Provider selection, fallback, capability validation, retries,
    observability, and provider execution remain owned by the Gateway.
    """

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]: ...


class UnavailableLLMGateway:
    """
    Explicit unavailable Gateway used when AgentRuntime is created
    without an LLM Gateway.
    """

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError("LLM Gateway is not configured for AgentRuntime.")


class AgentLLMContext:
    """
    Controlled LLM capability available during agent execution.

    The context contains the LLM configuration bound from the agent
    definition. Gateway responsibilities remain outside this class.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        config: AgentLLMConfig,
    ) -> None:
        self._gateway = gateway
        self._config = config

    @property
    def model(self) -> str | None:
        return self._config.model

    @property
    def system_prompt(self) -> str:
        return self._config.system_prompt

    @property
    def temperature(self) -> float:
        return self._config.temperature

    @property
    def max_tokens(self) -> int:
        return self._config.max_tokens

    def build_messages(
        self,
        *,
        prompt: str,
        history: tuple[AgentMessage, ...] = (),
        tool_results: tuple[str, ...] = (),
    ) -> tuple[AgentMessage, ...]:
        """
        Build the ordered agent-side LLM conversation.

        The bound system prompt is always the first message.
        Conversation history follows it, then the current user
        message, followed by any tool-result messages.
        """
        if not prompt.strip():
            raise ValueError("LLM prompt must not be empty.")

        for message in history:
            if not isinstance(message, AgentMessage):
                raise TypeError("LLM history must contain AgentMessage instances.")

        messages: list[AgentMessage] = [
            system_message(self.system_prompt),
        ]

        messages.extend(history)

        messages.append(
            user_message(prompt),
        )

        for result in tool_results:
            messages.append(
                tool_message(result),
            )

        return tuple(messages)

    @staticmethod
    def _serialize_messages(
        messages: tuple[AgentMessage, ...],
    ) -> list[dict[str, Any]]:
        """
        Convert agent messages into the Gateway's structured message format.

        Ordinary messages remain simple role/content pairs.

        Assistant tool-call messages are recognized from the canonical
        Agent-side JSON representation and translated into structured
        Gateway metadata.

        Canonical tool-result messages are similarly translated into
        Gateway tool metadata.

        Generic tool messages that are not canonical tool-result messages
        remain ordinary role/content messages.
        """

        for message in messages:
            if not isinstance(message, AgentMessage):
                raise TypeError("LLM messages must contain AgentMessage instances.")

        serialized: list[dict[str, Any]] = []

        for message in messages:
            # ---------------------------------------------------------
            # SYSTEM / USER
            # ---------------------------------------------------------
            #
            # Preserve the original simple Gateway message shape.
            #
            if message.role in {
                AgentMessageRole.SYSTEM,
                AgentMessageRole.USER,
            }:
                serialized.append(
                    {
                        "role": message.role.value,
                        "content": message.content,
                    }
                )
                continue

            # ---------------------------------------------------------
            # ASSISTANT
            # ---------------------------------------------------------
            if message.role == AgentMessageRole.ASSISTANT:
                try:
                    payload = json.loads(message.content)
                except json.JSONDecodeError:
                    # Ordinary assistant message.
                    serialized.append(
                        {
                            "role": message.role.value,
                            "content": message.content,
                        }
                    )
                    continue

                # Valid JSON but not an object means this is still
                # ordinary assistant content.
                if not isinstance(payload, dict):
                    serialized.append(
                        {
                            "role": message.role.value,
                            "content": message.content,
                        }
                    )
                    continue

                # Only messages containing "tool_calls" are treated as
                # structured assistant tool-call messages.
                if "tool_calls" not in payload:
                    serialized.append(
                        {
                            "role": message.role.value,
                            "content": message.content,
                        }
                    )
                    continue

                raw_tool_calls = payload.get("tool_calls")

                if not isinstance(raw_tool_calls, list):
                    raise ValueError("Assistant tool-call message tool_calls " "must be a list.")

                tool_calls: list[dict[str, Any]] = []

                for raw_tool_call in raw_tool_calls:
                    if not isinstance(raw_tool_call, dict):
                        raise ValueError(
                            "Assistant tool-call message tool_calls " "must contain dictionaries."
                        )

                    call_id = raw_tool_call.get("call_id")
                    name = raw_tool_call.get("name")
                    arguments = raw_tool_call.get(
                        "arguments",
                        {},
                    )

                    if not isinstance(call_id, str) or not call_id.strip():
                        raise ValueError(
                            "Assistant tool-call message call_id " "must be a non-empty string."
                        )

                    if not isinstance(name, str) or not name.strip():
                        raise ValueError(
                            "Assistant tool-call message name " "must be a non-empty string."
                        )

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Assistant tool-call message arguments " "must be a dictionary."
                        )

                    tool_calls.append(
                        {
                            "call_id": call_id,
                            "name": name,
                            "arguments": dict(arguments),
                        }
                    )

                content = payload.get(
                    "content",
                    "",
                )

                if not isinstance(content, str):
                    raise ValueError("Assistant tool-call message content " "must be a string.")

                if not content.strip():
                    content = "Tool call requested."

                serialized.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
                continue

            # ---------------------------------------------------------
            # TOOL
            # ---------------------------------------------------------
            if message.role == AgentMessageRole.TOOL:
                try:
                    payload = json.loads(message.content)
                except json.JSONDecodeError:
                    # Generic tool messages are valid and must remain
                    # ordinary tool messages.
                    serialized.append(
                        {
                            "role": "tool",
                            "content": message.content,
                        }
                    )
                    continue

                # Valid JSON but not an object means this is not a
                # canonical tool-result message.
                if not isinstance(payload, dict):
                    serialized.append(
                        {
                            "role": "tool",
                            "content": message.content,
                        }
                    )
                    continue

                # A generic JSON tool message that does not contain
                # canonical tool-result metadata remains unchanged.
                canonical_tool_result_keys = {
                    "call_id",
                    "tool_name",
                    "success",
                    "output",
                    "error",
                }

                if not (canonical_tool_result_keys & set(payload.keys())):
                    serialized.append(
                        {
                            "role": "tool",
                            "content": message.content,
                        }
                    )
                    continue

                # -----------------------------------------------------
                # Canonical tool-result validation
                # -----------------------------------------------------

                call_id = payload.get("call_id")
                tool_name = payload.get("tool_name")
                success = payload.get("success")

                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError("Tool result message call_id " "must be a non-empty string.")

                if not isinstance(tool_name, str) or not tool_name.strip():
                    raise ValueError("Tool result message tool_name " "must be a non-empty string.")

                if not isinstance(success, bool):
                    raise ValueError("Tool result message success " "must be a boolean.")

                if success:
                    if "output" not in payload:
                        raise ValueError("Successful tool result message " "must contain output.")
                else:
                    error = payload.get("error")

                    if not isinstance(error, str) or not error.strip():
                        raise ValueError(
                            "Failed tool result message error " "must be a non-empty string."
                        )

                serialized.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": call_id,
                        "tool_name": tool_name,
                    }
                )
                continue

            raise ValueError(f"Unsupported agent message role: {message.role!r}.")

        return serialized

    @staticmethod
    def _normalize_response(
        response: dict[str, Any],
    ) -> AgentLLMResult:
        """Convert a Gateway response into the Agent-facing result contract."""

        if not isinstance(response, dict):
            raise TypeError("LLM Gateway response must be a dictionary.")

        text = response.get("reply")

        if not isinstance(text, str):
            raise TypeError("LLM Gateway response reply must be a string.")

        provider = response.get("provider")

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("LLM Gateway response must contain a non-empty provider.")

        model = response.get("model")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("LLM Gateway response must contain a non-empty model.")

        usage = response.get("usage")

        if not isinstance(usage, dict):
            raise ValueError("LLM Gateway response must contain usage information.")

        prompt_tokens = usage.get(
            "prompt_tokens",
            0,
        )
        completion_tokens = usage.get(
            "completion_tokens",
            0,
        )
        total_tokens = usage.get(
            "total_tokens",
            0,
        )

        for name, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"LLM Gateway response {name} must be an integer.")

        raw_tool_calls = response.get(
            "tool_calls",
            (),
        )

        if raw_tool_calls is None:
            raw_tool_calls = ()

        if not isinstance(
            raw_tool_calls,
            (list, tuple),
        ):
            raise ValueError("LLM Gateway response tool_calls " "must be a list or tuple.")

        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, AgentToolCall):
                raise TypeError(
                    "LLM Gateway response tool_calls must contain " "AgentToolCall instances."
                )

        if not text.strip() and not raw_tool_calls:
            raise ValueError(
                "LLM Gateway response reply must not be empty " "when no tool calls are present."
            )

        return AgentLLMResult(
            text=text,
            provider=provider,
            model=model,
            usage=AgentLLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            raw_response=response,
            tool_calls=tuple(raw_tool_calls),
        )

    @staticmethod
    def _serialize_tools(
        tools: tuple[ToolDefinition, ...],
    ) -> list[dict[str, Any]]:
        """
        Convert agent-visible tool definitions into the provider-neutral
        Gateway tool schema.

        The Agent layer exposes only the tool contract. Provider-specific
        translation remains the responsibility of the LLM Gateway/provider.
        """
        for tool in tools:
            if not isinstance(tool, ToolDefinition):
                raise TypeError("LLM tools must contain ToolDefinition instances.")

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in tools
        ]

    async def generate(
        self,
        *,
        prompt: str,
        messages: tuple[AgentMessage, ...] | None = None,
        tools: tuple[ToolDefinition, ...] | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        user_id: str | None = None,
    ) -> AgentLLMResult:
        if not prompt.strip():
            raise ValueError("LLM prompt must not be empty.")

        resolved_model = model if model is not None else self._config.model

        if resolved_model is None or not resolved_model.strip():
            raise ValueError(
                "LLM model must be provided either by the agent "
                "configuration or the generate call."
            )

        resolved_temperature = self._config.temperature if temperature is None else temperature

        resolved_max_tokens = self._config.max_tokens if max_tokens is None else max_tokens

        if resolved_temperature < 0 or resolved_temperature > 2:
            raise ValueError("LLM temperature must be between 0 and 2.")

        if resolved_max_tokens <= 0:
            raise ValueError("LLM max_tokens must be greater than zero.")

        request: dict[str, Any] = {
            "prompt": prompt,
            "model": resolved_model,
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
            "stream": False,
        }

        if messages is not None:
            request["messages"] = self._serialize_messages(messages)

        if tools is not None:
            request["tools"] = self._serialize_tools(tools)

        if provider is not None:
            if not provider.strip():
                raise ValueError("LLM provider must not be empty when provided.")

            request["provider"] = provider

        if user_id is not None:
            if not user_id.strip():
                raise ValueError("LLM user_id must not be empty when provided.")

            request["user_id"] = user_id

        response = await self._gateway.route_chat(request)

        return self._normalize_response(response)
