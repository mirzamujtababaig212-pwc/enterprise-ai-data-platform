import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AuthenticationError,
    RateLimitError,
)

from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.llm_gateway.config.azure_openai_settings import (
    AzureOpenAISettings,
    get_azure_openai_settings,
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
    "azure-openai-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "azure-openai-embedding",
}


class AzureOpenAIProvider(BaseProvider):
    def __init__(
        self,
        client: AsyncAzureOpenAI | None = None,
        settings: AzureOpenAISettings | None = None,
    ) -> None:
        self.settings = settings or get_azure_openai_settings()

        self.client = client

        if self.client is None:
            if self.settings.api_key and self.settings.endpoint:
                self.client = AsyncAzureOpenAI(
                    api_key=self.settings.api_key,
                    azure_endpoint=self.settings.endpoint,
                    api_version=self.settings.api_version,
                    timeout=self.settings.timeout,
                    max_retries=self.settings.max_retries,
                )

    @staticmethod
    def _rate_limit_message(exc: RateLimitError) -> str:
        body = getattr(exc, "body", None)

        if isinstance(body, dict):
            error_body = body.get("error")

            if isinstance(error_body, dict):
                code = error_body.get("code")
                message = error_body.get("message")

                if code and message:
                    return f"Azure OpenAI error code: {code}. {message}"

                if code:
                    return f"Azure OpenAI error code: {code}."

                if message:
                    return str(message)

        return str(exc)

    @staticmethod
    def _is_quota_exceeded(exc: RateLimitError) -> bool:
        code = getattr(exc, "code", None)

        if isinstance(code, str) and code.lower() in {
            "insufficient_quota",
            "quota_exceeded",
        }:
            return True

        body = getattr(exc, "body", None)

        if isinstance(body, dict):
            error_body = body.get("error")

            if isinstance(error_body, dict):
                body_code = error_body.get("code")

                if isinstance(body_code, str) and body_code.lower() in {
                    "insufficient_quota",
                    "quota_exceeded",
                }:
                    return True

        return False

    @staticmethod
    def _messages(request: dict[str, Any]) -> list[dict[str, str]]:
        messages = request.get("messages")

        if messages is not None:
            if not isinstance(messages, list):
                raise ValueError("Azure OpenAI chat messages must be a list.")

            if not messages:
                raise ValueError("Azure OpenAI chat messages must not be empty.")

            normalized: list[dict[str, str]] = []

            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("Azure OpenAI chat messages must contain dictionaries.")

                role = message.get("role")
                content = message.get("content")

                if not isinstance(role, str) or not role.strip():
                    raise ValueError("Azure OpenAI chat message role must not be empty.")

                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Azure OpenAI chat message content must not be empty.")

                normalized.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

            return normalized

        prompt = request.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Azure OpenAI chat prompt must not be empty.")

        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    @staticmethod
    def _tools(
        request: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        raw_tools = request.get("tools")

        if raw_tools is None:
            return None

        if not isinstance(raw_tools, list):
            raise ValueError("Azure OpenAI chat tools must be a list.")

        tools: list[dict[str, Any]] = []

        for tool in raw_tools:
            if not isinstance(tool, dict):
                raise ValueError("Azure OpenAI chat tools must contain dictionaries.")

            if "function" in tool:
                function = tool["function"]

                if not isinstance(function, dict):
                    raise ValueError("Azure OpenAI function tool must contain a dictionary.")

                name = function.get("name")
                description = function.get("description", "")
                parameters = function.get("parameters", {})

            else:
                name = tool.get("name")
                description = tool.get("description", "")
                parameters = tool.get("input_schema", {})

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Azure OpenAI chat tool name must be a non-empty string.")

            if not isinstance(description, str):
                raise ValueError("Azure OpenAI chat tool description must be a string.")

            if not isinstance(parameters, dict):
                raise ValueError("Azure OpenAI chat tool parameters must be a dictionary.")

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": dict(parameters),
                    },
                }
            )

        return tools

    @staticmethod
    def _extract_tool_calls(
        response: Any,
    ) -> list[AgentToolCall]:
        choices = getattr(response, "choices", None)

        if not choices:
            return []

        message = getattr(choices[0], "message", None)

        if message is None:
            return []

        raw_tool_calls = getattr(message, "tool_calls", None)

        if not raw_tool_calls:
            return []

        tool_calls: list[AgentToolCall] = []

        for tool_call in raw_tool_calls:
            call_id = getattr(tool_call, "id", None)
            function = getattr(tool_call, "function", None)

            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("Azure OpenAI function call did not contain a valid call_id.")

            if function is None:
                raise ValueError("Azure OpenAI function call did not contain function data.")

            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Azure OpenAI function call did not contain a valid tool name.")

            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Azure OpenAI function call arguments were not valid JSON "
                        f"for tool '{name}'."
                    ) from exc
            elif isinstance(arguments, dict):
                parsed_arguments = arguments
            else:
                raise ValueError(
                    f"Azure OpenAI function call arguments were invalid " f"for tool '{name}'."
                )

            if not isinstance(parsed_arguments, dict):
                raise ValueError(
                    f"Azure OpenAI function call arguments must decode to an "
                    f"object for tool '{name}'."
                )

            tool_calls.append(
                AgentToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=parsed_arguments,
                )
            )

        return tool_calls

    @staticmethod
    def _deployment_for_chat(
        settings: AzureOpenAISettings,
    ) -> str:
        if not settings.chat_deployment:
            raise ProviderAuthenticationError("Azure OpenAI chat deployment is not configured.")

        return settings.chat_deployment

    @staticmethod
    def _deployment_for_embedding(
        settings: AzureOpenAISettings,
    ) -> str:
        if not settings.embedding_deployment:
            raise ProviderAuthenticationError(
                "Azure OpenAI embedding deployment is not configured."
            )

        return settings.embedding_deployment

    async def chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        model = request.get(
            "model",
            "azure-openai-chat",
        )

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Azure OpenAI chat model: {model}")

        if self.client is None:
            raise ProviderAuthenticationError("Azure OpenAI provider is not configured.")

        messages = self._messages(request)
        tools = self._tools(request)

        deployment = self._deployment_for_chat(self.settings)

        try:
            kwargs: dict[str, Any] = {
                "model": deployment,
                "messages": messages,
                "temperature": request.get("temperature", 0.7),
                "max_tokens": request.get("max_tokens", 1024),
            }

            if tools is not None:
                kwargs["tools"] = tools

            response = await self.client.chat.completions.create(**kwargs)

            choices = getattr(response, "choices", None)

            if not choices:
                raise ValueError("Azure OpenAI response contained no choices.")

            message = getattr(
                choices[0],
                "message",
                None,
            )

            if message is None:
                raise ValueError("Azure OpenAI response contained no message.")

            reply = getattr(
                message,
                "content",
                None,
            )

            if reply is None:
                reply = ""

            usage = getattr(response, "usage", None)

            tokens_in = (
                getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )
                if usage
                else 0
            )

            tokens_out = (
                getattr(
                    usage,
                    "completion_tokens",
                    0,
                )
                if usage
                else 0
            )

            return {
                "reply": reply,
                "usage": {
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
                "tool_calls": self._extract_tool_calls(response),
            }

        except AuthenticationError as exc:
            logger.exception("Azure OpenAI authentication failed.")
            raise ProviderAuthenticationError("Azure OpenAI authentication failed.") from exc

        except APITimeoutError as exc:
            logger.exception("Azure OpenAI request timed out.")
            raise ProviderTimeoutError("Azure OpenAI request timed out.") from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to Azure OpenAI.")
            raise ProviderConnectionError("Unable to connect to Azure OpenAI.") from exc

        except RateLimitError as exc:
            message = self._rate_limit_message(exc)

            if self._is_quota_exceeded(exc):
                logger.exception(
                    "Azure OpenAI quota exceeded: %s",
                    message,
                )
                raise ProviderQuotaExceededError(f"Azure OpenAI quota exceeded: {message}") from exc

            logger.exception(
                "Azure OpenAI rate limit exceeded: %s",
                message,
            )
            raise ProviderRateLimitError(f"Azure OpenAI rate limit exceeded: {message}") from exc

        except (
            ProviderAuthenticationError,
            ProviderConnectionError,
            ProviderExecutionError,
            ProviderQuotaExceededError,
            ProviderRateLimitError,
            ProviderTimeoutError,
        ):
            raise

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Unexpected Azure OpenAI provider error.")
            raise ProviderExecutionError("Unexpected Azure OpenAI provider error.") from exc

    async def stream(
        self,
        request: dict[str, Any],
    ) -> AsyncIterator[str]:
        model = request.get(
            "model",
            "azure-openai-chat",
        )

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Azure OpenAI chat model: {model}")

        if self.client is None:
            raise ProviderAuthenticationError("Azure OpenAI provider is not configured.")

        messages = self._messages(request)
        deployment = self._deployment_for_chat(self.settings)

        try:
            stream = await self.client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=request.get("temperature", 0.7),
                max_tokens=request.get("max_tokens", 1024),
                stream=True,
            )

            async for chunk in stream:
                choices = getattr(chunk, "choices", None)

                if not choices:
                    continue

                delta = getattr(
                    choices[0],
                    "delta",
                    None,
                )

                if delta is None:
                    continue

                content = getattr(
                    delta,
                    "content",
                    None,
                )

                if content:
                    yield content

        except AuthenticationError as exc:
            logger.exception("Azure OpenAI streaming authentication failed.")
            raise ProviderAuthenticationError("Azure OpenAI authentication failed.") from exc

        except APITimeoutError as exc:
            logger.exception("Azure OpenAI streaming request timed out.")
            raise ProviderTimeoutError("Azure OpenAI request timed out.") from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to Azure OpenAI during streaming.")
            raise ProviderConnectionError("Unable to connect to Azure OpenAI.") from exc

        except RateLimitError as exc:
            message = self._rate_limit_message(exc)

            if self._is_quota_exceeded(exc):
                raise ProviderQuotaExceededError(f"Azure OpenAI quota exceeded: {message}") from exc

            raise ProviderRateLimitError(f"Azure OpenAI rate limit exceeded: {message}") from exc

        except (
            ProviderAuthenticationError,
            ProviderConnectionError,
            ProviderExecutionError,
            ProviderQuotaExceededError,
            ProviderRateLimitError,
            ProviderTimeoutError,
        ):
            raise

        except Exception as exc:
            logger.exception("Unexpected Azure OpenAI streaming error.")
            raise ProviderConnectionError("Unexpected Azure OpenAI streaming error.") from exc

    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        model = request.get("model")

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Azure OpenAI embedding model: {model}")

        text = request.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input text must not be empty.")

        if self.client is None:
            raise ProviderAuthenticationError("Azure OpenAI provider is not configured.")

        deployment = self._deployment_for_embedding(self.settings)

        try:
            response = await self.client.embeddings.create(
                model=deployment,
                input=text,
            )

            data = getattr(response, "data", None)

            if not data:
                raise ValueError("Azure OpenAI embedding response contained no data.")

            embedding = getattr(
                data[0],
                "embedding",
                None,
            )

            if not embedding:
                raise ValueError("Azure OpenAI embedding response contained an empty vector.")

            return embedding

        except AuthenticationError as exc:
            logger.exception("Azure OpenAI embedding authentication failed.")
            raise ProviderAuthenticationError("Azure OpenAI authentication failed.") from exc

        except RateLimitError as exc:
            message = self._rate_limit_message(exc)

            if self._is_quota_exceeded(exc):
                raise ProviderQuotaExceededError(f"Azure OpenAI quota exceeded: {message}") from exc

            raise ProviderRateLimitError(f"Azure OpenAI rate limit exceeded: {message}") from exc

        except APITimeoutError as exc:
            logger.exception("Azure OpenAI embedding request timed out.")
            raise ProviderTimeoutError("Azure OpenAI request timed out.") from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to Azure OpenAI embedding API.")
            raise ProviderConnectionError("Unable to connect to Azure OpenAI.") from exc

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Unexpected Azure OpenAI embedding error.")
            raise ProviderExecutionError("Unexpected Azure OpenAI embedding error.") from exc

    async def health_check(
        self,
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "configured": bool(self.settings.api_key and self.settings.endpoint),
            "endpoint": self.settings.endpoint,
            "api_version": self.settings.api_version,
            "chat_deployment": self.settings.chat_deployment,
            "embedding_deployment": self.settings.embedding_deployment,
        }

    async def list_models(
        self,
    ) -> list[str]:
        return [
            *sorted(SUPPORTED_CHAT_MODELS),
            *sorted(SUPPORTED_EMBEDDING_MODELS),
        ]

    def supported_chat_models(
        self,
    ) -> list[str]:
        return sorted(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(
        self,
    ) -> list[str]:
        return sorted(SUPPORTED_EMBEDDING_MODELS)

    def supported_stream_models(
        self,
    ) -> list[str]:
        return sorted(SUPPORTED_CHAT_MODELS)
