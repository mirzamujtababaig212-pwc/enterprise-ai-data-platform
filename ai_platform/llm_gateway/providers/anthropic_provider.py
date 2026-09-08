from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic
from anthropic import APIConnectionError
from anthropic import APIError
from anthropic import APITimeoutError
from anthropic import AuthenticationError
from anthropic import RateLimitError

from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.llm_gateway.providers.base_provider import BaseProvider

SUPPORTED_CHAT_MODELS = {
    "anthropic-chat",
}

SUPPORTED_EMBEDDING_MODELS: set[str] = set()

ANTHROPIC_CHAT_MODEL_MAP = {
    "anthropic-chat": "claude-sonnet-4-6",
}


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        api_key: str | None = None,
    ):
        self.client = client

        if self.client is None:
            if api_key is None:
                from ai_platform.llm_gateway.config.settings import settings

                api_key = settings.ANTHROPIC_API_KEY

            if api_key:
                self.client = AsyncAnthropic(api_key=api_key)

    def _require_client(self) -> AsyncAnthropic:
        if self.client is None:
            raise AuthenticationError(
                message="Anthropic API key is not configured",
                response=None,
                body=None,
            )
        return self.client

    @staticmethod
    def _physical_model(model: str) -> str:
        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Anthropic chat model: {model}")

        return ANTHROPIC_CHAT_MODEL_MAP[model]

    @staticmethod
    def _messages(request: dict[str, Any]) -> list[dict[str, Any]]:
        messages = request.get("messages")

        if messages is None:
            prompt = request.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Anthropic chat requires a non-empty prompt or messages list.")
            return [{"role": "user", "content": prompt}]

        if not isinstance(messages, list) or not messages:
            raise ValueError("Anthropic messages must be a non-empty list.")

        result: list[dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Anthropic messages must contain dictionaries.")

            role = message.get("role")
            content = message.get("content")

            if role == "system":
                continue

            if role not in {"user", "assistant"}:
                raise ValueError("Anthropic messages only support user and assistant roles.")

            if not isinstance(content, (str, list)):
                raise ValueError("Anthropic message content must be a string or content list.")

            result.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        if not result:
            raise ValueError("Anthropic messages must contain user or assistant messages.")

        return result

    @staticmethod
    def _system(request: dict[str, Any]) -> str | None:
        messages = request.get("messages")

        if not isinstance(messages, list):
            return None

        system_messages = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            if message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    system_messages.append(content)

        return "\n\n".join(system_messages) or None

    @staticmethod
    def _tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
        raw_tools = request.get("tools")

        if raw_tools is None:
            return None

        if not isinstance(raw_tools, list):
            raise ValueError("Anthropic tools must be a list.")

        tools: list[dict[str, Any]] = []

        for tool in raw_tools:
            if not isinstance(tool, dict):
                raise ValueError("Anthropic tools must contain dictionaries.")

            function = tool.get("function", tool)

            if not isinstance(function, dict):
                raise ValueError("Anthropic tool definition must be a dictionary.")

            name = function.get("name")
            description = function.get("description", "")
            input_schema = function.get(
                "input_schema",
                function.get("parameters", {}),
            )

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Anthropic tool name must be a non-empty string.")

            if not isinstance(description, str):
                raise ValueError("Anthropic tool description must be a string.")

            if not isinstance(input_schema, dict):
                raise ValueError("Anthropic tool input schema must be a dictionary.")

            tools.append(
                {
                    "name": name,
                    "description": description,
                    "input_schema": input_schema,
                }
            )

        return tools

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[AgentToolCall]:
        tool_calls: list[AgentToolCall] = []

        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) != "tool_use":
                continue

            name = getattr(block, "name", None)
            tool_input = getattr(block, "input", None)

            if not isinstance(name, str) or not name:
                raise ValueError("Anthropic tool call did not contain a valid tool name.")

            if not isinstance(tool_input, dict):
                raise ValueError(f"Anthropic tool call arguments were invalid for tool '{name}'.")

            call_id = getattr(block, "id", None) or f"anthropic-{len(tool_calls) + 1}"

            tool_calls.append(
                AgentToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=tool_input,
                )
            )

        return tool_calls

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)

        if usage is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    @staticmethod
    def _raise_provider_error(exc: Exception) -> None:
        if isinstance(exc, AuthenticationError):
            raise RuntimeError(f"Anthropic authentication error: {exc}") from exc

        if isinstance(exc, RateLimitError):
            raise RuntimeError(f"Anthropic rate limit error: {exc}") from exc

        if isinstance(exc, APITimeoutError):
            raise TimeoutError(f"Anthropic request timed out: {exc}") from exc

        if isinstance(exc, APIConnectionError):
            raise ConnectionError(f"Anthropic connection error: {exc}") from exc

        if isinstance(exc, APIError):
            raise RuntimeError(f"Anthropic API error: {exc}") from exc

        raise RuntimeError(f"Anthropic execution error: {exc}") from exc

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        client = self._require_client()

        model = self._physical_model(request.get("model", "anthropic-chat"))
        messages = self._messages(request)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": int(request.get("max_tokens", 1024)),
            "messages": messages,
        }

        system = self._system(request)
        if system is not None:
            kwargs["system"] = system

        tools = self._tools(request)
        if tools is not None:
            kwargs["tools"] = tools

        try:
            response = await client.messages.create(**kwargs)

            text_parts = [
                block.text
                for block in getattr(response, "content", []) or []
                if getattr(block, "type", None) == "text"
                and isinstance(getattr(block, "text", None), str)
            ]

            tool_calls = self._extract_tool_calls(response)

            return {
                "reply": "".join(text_parts),
                "usage": self._usage(response),
                "tool_calls": tool_calls,
            }

        except Exception as exc:
            self._raise_provider_error(exc)
            raise AssertionError("unreachable")

    async def stream(self, request: dict[str, Any]) -> AsyncIterator[str]:
        client = self._require_client()

        model = self._physical_model(request.get("model", "anthropic-chat"))
        messages = self._messages(request)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": int(request.get("max_tokens", 1024)),
            "messages": messages,
        }

        system = self._system(request)
        if system is not None:
            kwargs["system"] = system

        tools = self._tools(request)
        if tools is not None:
            kwargs["tools"] = tools

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            self._raise_provider_error(exc)

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        model = request.get("model")

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                "Anthropic does not provide an embeddings capability for this provider."
            )

        raise ValueError(f"Unsupported Anthropic embedding model: {model}")

    async def health_check(self) -> dict[str, Any]:
        if self.client is None:
            return {
                "status": "unconfigured",
                "provider": "anthropic",
            }

        return {
            "status": "configured",
            "provider": "anthropic",
        }

    async def list_models(self) -> list[str]:
        return list(SUPPORTED_CHAT_MODELS)

    def supported_chat_models(self) -> list[str]:
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self) -> list[str]:
        return list(SUPPORTED_EMBEDDING_MODELS)

    def supported_stream_models(self) -> list[str]:
        return list(SUPPORTED_CHAT_MODELS)
