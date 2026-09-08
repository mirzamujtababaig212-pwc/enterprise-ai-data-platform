import logging
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.llm_gateway.config.gemini_settings import (
    GeminiSettings,
    get_gemini_settings,
)
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderExecutionError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ai_platform.llm_gateway.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


SUPPORTED_CHAT_MODELS = {
    "gemini-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "gemini-embedding",
}

GEMINI_CHAT_MODEL_MAP = {
    "gemini-chat": "gemini-2.5-flash",
}

GEMINI_EMBEDDING_MODEL_MAP = {
    "gemini-embedding": "gemini-embedding-001",
}


class GeminiProvider(BaseProvider):
    def __init__(
        self,
        client: genai.Client | None = None,
        settings: GeminiSettings | None = None,
    ) -> None:
        self.settings = settings or get_gemini_settings()

        self.default_model = "gemini-chat"

        self.client = client

        if self.client is None and self.settings.api_key:
            self.client = genai.Client(
                api_key=self.settings.api_key,
            )

    def _require_client(self) -> genai.Client:
        if self.client is None:
            raise ProviderAuthenticationError("Gemini provider is not configured.")

        return self.client

    @staticmethod
    def _chat_input(
        request: dict[str, Any],
    ) -> str | list[dict[str, Any]]:
        messages = request.get("messages")

        if messages is not None:
            if not isinstance(messages, list):
                raise ValueError("Gemini chat messages must be a list.")

            if not messages:
                raise ValueError("Gemini chat messages must not be empty.")

            normalized: list[dict[str, Any]] = []

            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("Gemini chat messages must contain dictionaries.")

                role = message.get("role")
                content = message.get("content")

                if not isinstance(role, str) or not role.strip():
                    raise ValueError("Gemini chat message role must not be empty.")

                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Gemini chat message content must not be empty.")

                normalized.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

            return normalized

        prompt = request.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Gemini chat prompt must not be empty.")

        return prompt

    @staticmethod
    def _contents(
        chat_input: str | list[dict[str, Any]],
    ) -> str | list[types.Content]:
        if isinstance(chat_input, str):
            return chat_input

        contents: list[types.Content] = []

        for message in chat_input:
            role = message["role"]

            # Gemini uses "user" and "model" roles.
            if role == "assistant":
                role = "model"

            if role == "system":
                # System messages are handled separately by the caller.
                continue

            contents.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(
                            text=message["content"],
                        )
                    ],
                )
            )

        if not contents:
            raise ValueError("Gemini chat messages must contain at least one user/model message.")

        return contents

    @staticmethod
    def _system_instruction(
        chat_input: str | list[dict[str, Any]],
    ) -> str | None:
        if isinstance(chat_input, str):
            return None

        system_messages = [
            message["content"] for message in chat_input if message["role"] == "system"
        ]

        if not system_messages:
            return None

        return "\n\n".join(system_messages)

    @staticmethod
    def _tool_config(
        request: dict[str, Any],
    ) -> types.GenerateContentConfig | None:
        tools = request.get("tools")

        if tools is None:
            return None

        if not isinstance(tools, list):
            raise ValueError("Gemini tools must be a list.")

        declarations = []

        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError("Gemini tools must contain dictionaries.")

            function = tool.get("function", tool)

            if not isinstance(function, dict):
                raise ValueError("Gemini tool function definition must be a dictionary.")

            name = function.get("name")
            description = function.get("description", "")
            parameters = function.get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                },
            )

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Gemini tool name must not be empty.")

            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )

        if not declarations:
            return None

        return types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=declarations,
                )
            ]
        )

    @staticmethod
    def _extract_tool_calls(
        response: Any,
    ) -> list[AgentToolCall]:
        tool_calls: list[AgentToolCall] = []

        candidates = getattr(response, "candidates", None)

        if not candidates:
            return tool_calls

        for candidate in candidates:
            content = getattr(candidate, "content", None)

            if content is None:
                continue

            parts = getattr(content, "parts", None)

            if not parts:
                continue

            for part in parts:
                function_call = getattr(part, "function_call", None)

                if function_call is None:
                    continue

                name = getattr(function_call, "name", None)
                arguments = getattr(function_call, "args", None)

                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Gemini function call did not contain a valid tool name.")

                if arguments is None:
                    arguments = {}

                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"Gemini function call arguments were invalid " f"for tool '{name}'."
                    )

                call_id = getattr(
                    function_call,
                    "id",
                    None,
                )

                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = f"gemini-{len(tool_calls) + 1}"

                tool_calls.append(
                    AgentToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=dict(arguments),
                    )
                )

        return tool_calls

    @staticmethod
    def _error_message(exc: genai_errors.APIError) -> str:
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", None)

        if code is not None and message:
            return f"Gemini API error {code}: {message}"

        if message:
            return str(message)

        return str(exc)

    @classmethod
    def _raise_provider_error(
        cls,
        exc: Exception,
    ) -> None:
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            message = cls._error_message(exc)

            if code in {401, 403}:
                raise ProviderAuthenticationError(message) from exc

            if code == 429:
                lowered = message.lower()

                if any(
                    token in lowered
                    for token in (
                        "quota",
                        "resource exhausted",
                        "exceeded",
                    )
                ):
                    raise ProviderQuotaExceededError(message) from exc

                raise ProviderRateLimitError(message) from exc

            if code in {408, 504}:
                raise ProviderTimeoutError(message) from exc

            if code is not None and code >= 500:
                raise ProviderConnectionError(message) from exc

            raise ProviderExecutionError(message) from exc

        if isinstance(exc, TimeoutError):
            raise ProviderTimeoutError("Gemini request timed out.") from exc

        if isinstance(exc, OSError):
            raise ProviderConnectionError("Unable to connect to Gemini.") from exc

        raise ProviderExecutionError("Unexpected Gemini provider error.") from exc

    async def chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        chat_input = self._chat_input(request)

        model = request.get(
            "model",
            "gemini-chat",
        )

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Gemini chat model: {model}")

        client = self._require_client()

        contents = self._contents(chat_input)
        system_instruction = self._system_instruction(chat_input)

        config_kwargs: dict[str, Any] = {}

        temperature = request.get("temperature")
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        max_tokens = request.get("max_tokens")
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        tool_config = self._tool_config(request)

        if tool_config is not None:
            config_kwargs["tools"] = tool_config.tools

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_CHAT_MODEL_MAP[model],
                contents=contents,
                config=config,
            )

            reply = getattr(response, "text", None)

            if reply is None:
                reply = ""

            usage_metadata = getattr(
                response,
                "usage_metadata",
                None,
            )

            tokens_in = (
                getattr(
                    usage_metadata,
                    "prompt_token_count",
                    0,
                )
                or 0
            )

            tokens_out = (
                getattr(
                    usage_metadata,
                    "candidates_token_count",
                    0,
                )
                or 0
            )

            return {
                "reply": reply,
                "usage": {
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
                "tool_calls": self._extract_tool_calls(response),
            }

        except (
            ProviderAuthenticationError,
            ProviderRateLimitError,
            ProviderQuotaExceededError,
            ProviderTimeoutError,
            ProviderConnectionError,
            ProviderExecutionError,
        ):
            raise

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Gemini chat request failed.")
            self._raise_provider_error(exc)

        raise ProviderExecutionError("Unexpected Gemini chat failure.")

    async def stream(
        self,
        request: dict[str, Any],
    ) -> AsyncIterator[str]:
        chat_input = self._chat_input(request)

        model = request.get(
            "model",
            "gemini-chat",
        )

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Gemini chat model: {model}")

        client = self._require_client()

        contents = self._contents(chat_input)
        system_instruction = self._system_instruction(chat_input)

        config_kwargs: dict[str, Any] = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        max_tokens = request.get("max_tokens")
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            stream = await client.aio.models.generate_content_stream(
                model=GEMINI_CHAT_MODEL_MAP[model],
                contents=contents,
                config=config,
            )

            async for chunk in stream:
                text = getattr(chunk, "text", None)

                if text:
                    yield text

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Gemini streaming request failed.")
            self._raise_provider_error(exc)

    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        model = request.get("model")
        text = request.get("text")

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Gemini embedding model: {model}")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input text must not be empty.")

        client = self._require_client()

        try:
            response = await client.aio.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL_MAP[model],
                contents=text,
            )

            embeddings = getattr(
                response,
                "embeddings",
                None,
            )

            if not embeddings:
                raise ValueError("Gemini embedding response contained no data.")

            embedding = getattr(
                embeddings[0],
                "values",
                None,
            )

            if not embedding:
                raise ValueError("Gemini embedding response contained an empty vector.")

            return list(embedding)

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Gemini embedding request failed.")
            self._raise_provider_error(exc)

        raise ProviderExecutionError("Unexpected Gemini embedding failure.")

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "configured": bool(self.settings.api_key),
            "default_model": self.default_model,
            "embedding_model": self.settings.embedding_model,
        }

    async def list_models(self) -> list[str]:
        return [
            *sorted(SUPPORTED_CHAT_MODELS),
            *sorted(SUPPORTED_EMBEDDING_MODELS),
        ]

    def supported_chat_models(self) -> list[str]:
        return sorted(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self) -> list[str]:
        return sorted(SUPPORTED_EMBEDDING_MODELS)

    def supported_stream_models(self) -> list[str]:
        return sorted(SUPPORTED_CHAT_MODELS)
