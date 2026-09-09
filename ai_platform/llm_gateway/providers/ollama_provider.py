import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.llm_gateway.config.ollama_settings import (
    OllamaSettings,
    get_ollama_settings,
)
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderExecutionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ai_platform.llm_gateway.providers.base_provider import BaseProvider

SUPPORTED_CHAT_MODELS = {
    "ollama-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "ollama-embedding",
}

OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_EMBEDDING_ENDPOINT = "/api/embed"
OLLAMA_TAGS_ENDPOINT = "/api/tags"


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        settings: OllamaSettings | None = None,
    ) -> None:
        self.settings = settings or get_ollama_settings()

        self.client = client

        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=self.settings.timeout,
            )

    @staticmethod
    def _messages(request: dict[str, Any]) -> list[dict[str, str]]:
        messages = request.get("messages")

        if messages is None:
            prompt = request.get("prompt")

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Ollama chat requires a non-empty prompt or messages list.")

            return [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        if not isinstance(messages, list) or not messages:
            raise ValueError("Ollama messages must be a non-empty list.")

        normalized: list[dict[str, str]] = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Ollama messages must contain dictionaries.")

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str) or not role.strip():
                raise ValueError("Ollama message role must not be empty.")

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Ollama message content must not be empty.")

            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return normalized

    def _physical_chat_model(self, model: str) -> str:
        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Ollama chat model: {model}")

        if not self.settings.chat_model.strip():
            raise ValueError("Ollama chat model is not configured.")

        return self.settings.chat_model

    def _physical_embedding_model(self, model: str) -> str:
        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Ollama embedding model: {model}")

        if not self.settings.embedding_model.strip():
            raise ValueError("Ollama embedding model is not configured.")

        return self.settings.embedding_model

    @staticmethod
    def _raise_http_error(response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise ProviderAuthenticationError("Ollama authentication failed.")

        if response.status_code == 429:
            raise ProviderRateLimitError("Ollama rate limit exceeded.")

        if response.status_code >= 500:
            raise ProviderExecutionError(f"Ollama server error: HTTP {response.status_code}.")

        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text

            if isinstance(detail, dict):
                message = detail.get("error")
            else:
                message = detail

            raise ProviderExecutionError(f"Ollama request failed: {message}")

    @staticmethod
    def _usage(response: dict[str, Any]) -> dict[str, int]:
        return {
            "tokens_in": int(response.get("prompt_eval_count", 0) or 0),
            "tokens_out": int(response.get("eval_count", 0) or 0),
        }

    @staticmethod
    def _tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
        raw_tools = request.get("tools")

        if raw_tools is None:
            return None

        if not isinstance(raw_tools, list):
            raise ValueError("Ollama tools must be a list.")

        tools: list[dict[str, Any]] = []

        for tool in raw_tools:
            if not isinstance(tool, dict):
                raise ValueError("Ollama tools must contain dictionaries.")

            function = tool.get("function", tool)

            if not isinstance(function, dict):
                raise ValueError("Ollama tool definition must be a dictionary.")

            name = function.get("name")
            description = function.get("description", "")
            parameters = function.get(
                "parameters",
                function.get("input_schema", {}),
            )

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Ollama tool name must be a non-empty string.")

            if not isinstance(description, str):
                raise ValueError("Ollama tool description must be a string.")

            if not isinstance(parameters, dict):
                raise ValueError("Ollama tool parameters must be a dictionary.")

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )

        return tools

    @staticmethod
    def _extract_tool_calls(message: dict[str, Any]) -> list[AgentToolCall]:
        raw_tool_calls = message.get("tool_calls", [])

        if raw_tool_calls is None:
            return []

        if not isinstance(raw_tool_calls, list):
            raise ProviderExecutionError("Ollama tool_calls must be a list.")

        tool_calls: list[AgentToolCall] = []

        for index, raw_tool_call in enumerate(raw_tool_calls, start=1):
            if not isinstance(raw_tool_call, dict):
                raise ProviderExecutionError("Ollama tool call must be a dictionary.")

            function = raw_tool_call.get("function")

            if not isinstance(function, dict):
                raise ProviderExecutionError("Ollama tool call did not contain a function.")

            name = function.get("name")
            arguments = function.get("arguments", {})

            if not isinstance(name, str) or not name.strip():
                raise ProviderExecutionError("Ollama tool call did not contain a valid tool name.")

            if arguments is None:
                arguments = {}

            if not isinstance(arguments, dict):
                raise ProviderExecutionError(
                    f"Ollama tool call arguments were invalid for tool '{name}'."
                )

            call_id = raw_tool_call.get("id")

            if not isinstance(call_id, str) or not call_id.strip():
                call_id = f"ollama-{index}"

            tool_calls.append(
                AgentToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=dict(arguments),
                )
            )

        return tool_calls

    async def chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        model = self._physical_chat_model(request.get("model", "ollama-chat"))

        messages = self._messages(request)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if "temperature" in request:
            payload["options"] = {
                "temperature": request["temperature"],
            }

        tools = self._tools(request)

        if tools is not None:
            payload["tools"] = tools

        try:
            response = await self.client.post(
                OLLAMA_CHAT_ENDPOINT,
                json=payload,
            )

            self._raise_http_error(response)

            body = response.json()

            if not isinstance(body, dict):
                raise ProviderExecutionError("Ollama chat response was not a JSON object.")

            message = body.get("message")

            if not isinstance(message, dict):
                raise ProviderExecutionError("Ollama chat response did not contain a message.")

            reply = message.get("content", "")

            if not isinstance(reply, str):
                raise ProviderExecutionError("Ollama chat response contained invalid content.")

            return {
                "reply": reply,
                "usage": self._usage(body),
                "tool_calls": self._extract_tool_calls(message),
            }

        except ProviderExecutionError:
            raise
        except (
            ProviderAuthenticationError,
            ProviderConnectionError,
            ProviderRateLimitError,
            ProviderTimeoutError,
        ):
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Ollama request timed out.") from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError("Unable to connect to Ollama.") from exc
        except httpx.HTTPError as exc:
            raise ProviderConnectionError("Ollama HTTP request failed.") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderExecutionError("Ollama returned an invalid response.") from exc
        except Exception as exc:
            raise ProviderExecutionError("Unexpected Ollama provider error.") from exc

    async def stream(
        self,
        request: dict[str, Any],
    ) -> AsyncIterator[str]:
        model = self._physical_chat_model(request.get("model", "ollama-chat"))

        messages = self._messages(request)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if "temperature" in request:
            payload["options"] = {
                "temperature": request["temperature"],
            }

        tools = self._tools(request)

        if tools is not None:
            payload["tools"] = tools

        try:
            async with self.client.stream(
                "POST",
                OLLAMA_CHAT_ENDPOINT,
                json=payload,
            ) as response:
                self._raise_http_error(response)

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        body = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderExecutionError(
                            "Ollama stream returned invalid JSON."
                        ) from exc

                    if not isinstance(body, dict):
                        continue

                    message = body.get("message")

                    if isinstance(message, dict):
                        content = message.get("content")

                        if isinstance(content, str) and content:
                            yield content

        except ProviderExecutionError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Ollama streaming request timed out.") from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError("Unable to connect to Ollama.") from exc
        except httpx.HTTPError as exc:
            raise ProviderConnectionError("Ollama streaming request failed.") from exc
        except Exception as exc:
            raise ProviderExecutionError("Unexpected Ollama streaming error.") from exc

    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        model = self._physical_embedding_model(request.get("model", "ollama-embedding"))

        text = request.get("input", request.get("text"))

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Ollama embeddings require non-empty input text.")

        payload = {
            "model": model,
            "input": text,
        }

        try:
            response = await self.client.post(
                OLLAMA_EMBEDDING_ENDPOINT,
                json=payload,
            )

            self._raise_http_error(response)

            body = response.json()

            if not isinstance(body, dict):
                raise ProviderExecutionError("Ollama embedding response was not a JSON object.")

            embeddings = body.get("embeddings")

            if not isinstance(embeddings, list) or not embeddings:
                raise ProviderExecutionError(
                    "Ollama embedding response did not contain embeddings."
                )

            vector = embeddings[0]

            if not isinstance(vector, list):
                raise ProviderExecutionError(
                    "Ollama embedding response contained an invalid vector."
                )

            return [float(value) for value in vector]

        except ProviderExecutionError:
            raise
        except (
            ProviderAuthenticationError,
            ProviderConnectionError,
            ProviderRateLimitError,
            ProviderTimeoutError,
        ):
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Ollama embedding request timed out.") from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError("Unable to connect to Ollama.") from exc
        except httpx.HTTPError as exc:
            raise ProviderConnectionError("Ollama embedding request failed.") from exc
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderExecutionError("Ollama returned an invalid embedding response.") from exc
        except Exception as exc:
            raise ProviderExecutionError("Unexpected Ollama embedding error.") from exc

    async def health_check(self) -> dict[str, Any]:
        try:
            response = await self.client.get(OLLAMA_TAGS_ENDPOINT)

            self._raise_http_error(response)

            body = response.json()

            models = body.get("models", []) if isinstance(body, dict) else []

            return {
                "status": "ok",
                "configured": True,
                "base_url": self.settings.base_url,
                "chat_model": self.settings.chat_model,
                "embedding_model": self.settings.embedding_model,
                "models_available": len(models),
            }

        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Ollama health check timed out.") from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError("Unable to connect to Ollama.") from exc
        except httpx.HTTPError as exc:
            raise ProviderConnectionError("Ollama health check failed.") from exc
        except Exception as exc:
            raise ProviderExecutionError("Unexpected Ollama health-check error.") from exc

    async def list_models(self) -> list[str]:
        response = await self.client.get(OLLAMA_TAGS_ENDPOINT)

        self._raise_http_error(response)

        body = response.json()

        if not isinstance(body, dict):
            raise ProviderExecutionError("Ollama model response was not a JSON object.")

        models = body.get("models", [])

        if not isinstance(models, list):
            raise ProviderExecutionError("Ollama model response contained invalid models.")

        result = []

        for model in models:
            if not isinstance(model, dict):
                continue

            name = model.get("name")

            if isinstance(name, str) and name.strip():
                result.append(name)

        return result

    def supported_chat_models(self) -> list[str]:
        return sorted(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self) -> list[str]:
        return sorted(SUPPORTED_EMBEDDING_MODELS)

    def supported_stream_models(self) -> list[str]:
        return self.supported_chat_models()
