import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from ai_platform.llm_gateway.config.bedrock_settings import (
    BedrockSettings,
    get_bedrock_settings,
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
    "bedrock-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "bedrock-embedding",
}


class BedrockProvider(BaseProvider):
    name = "bedrock"

    def __init__(
        self,
        client: Any | None = None,
        settings: BedrockSettings | None = None,
    ) -> None:
        self.settings = settings or get_bedrock_settings()

        self.client = client

        if self.client is None:
            if not self.settings.access_key_id or not self.settings.secret_access_key:
                self.client = None
            else:
                self.client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.settings.region,
                    aws_access_key_id=self.settings.access_key_id,
                    aws_secret_access_key=self.settings.secret_access_key,
                    config=boto3.session.Config(
                        connect_timeout=self.settings.timeout,
                        read_timeout=self.settings.timeout,
                    ),
                )

    def _require_client(self) -> Any:
        if self.client is None:
            raise ProviderAuthenticationError("Bedrock provider is not configured.")

        return self.client

    @staticmethod
    def _messages_from_request(
        request: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        messages = request.get("messages")

        if messages is None:
            prompt = request.get("prompt")

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Bedrock chat prompt must not be empty.")

            return (
                [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                [],
            )

        if not isinstance(messages, list) or not messages:
            raise ValueError("Bedrock chat messages must be a non-empty list.")

        conversation: list[dict[str, Any]] = []
        system: list[dict[str, str]] = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Bedrock chat messages must contain dictionaries.")

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str) or not role.strip():
                raise ValueError("Bedrock chat message role must not be empty.")

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Bedrock chat message content must not be empty.")

            if role == "system":
                system.append({"text": content})
            elif role in {"user", "assistant"}:
                conversation.append(
                    {
                        "role": role,
                        "content": [{"text": content}],
                    }
                )
            else:
                raise ValueError(f"Unsupported Bedrock message role: {role}")

        if not conversation:
            raise ValueError("Bedrock chat requires at least one user or assistant message.")

        return conversation, system

    @staticmethod
    def _error_message(exc: ClientError) -> str:
        error = exc.response.get("Error", {})
        code = error.get("Code", "UnknownError")
        message = error.get("Message", str(exc))
        return f"{code}: {message}"

    @classmethod
    def _raise_provider_error(cls, exc: Exception) -> None:
        if isinstance(exc, (EndpointConnectionError,)):
            raise ProviderConnectionError("Unable to connect to AWS Bedrock.") from exc

        if isinstance(exc, ReadTimeoutError):
            raise ProviderTimeoutError("AWS Bedrock request timed out.") from exc

        if isinstance(exc, ClientError):
            error = exc.response.get("Error", {})
            code = str(error.get("Code", "")).lower()
            message = cls._error_message(exc)

            if code in {
                "accessdeniedexception",
                "unrecognizedclientexception",
                "invalidsignatureexception",
                "expiredtokenexception",
                "invalidclienttokenid",
            }:
                raise ProviderAuthenticationError(
                    f"Bedrock authentication failed: {message}"
                ) from exc

            if code in {
                "throttlingexception",
                "toomanyrequestsexception",
            }:
                raise ProviderRateLimitError(f"Bedrock rate limit exceeded: {message}") from exc

            if code in {
                "servicequotafeatureexceededexception",
                "servicequotaexceededexception",
                "quotaexceededexception",
                "limitexceededexception",
            }:
                raise ProviderQuotaExceededError(f"Bedrock quota exceeded: {message}") from exc

            raise ProviderExecutionError(f"Bedrock request failed: {message}") from exc

        if isinstance(exc, BotoCoreError):
            raise ProviderConnectionError(f"Bedrock SDK error: {exc}") from exc

        raise ProviderExecutionError("Unexpected Bedrock provider error.") from exc

    async def chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        model = request.get("model", "bedrock-chat")

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Bedrock chat model: {model}")

        client = self._require_client()

        messages, system = self._messages_from_request(request)

        inference_config = {
            "maxTokens": int(request.get("max_tokens", 1024)),
            "temperature": float(request.get("temperature", 0.7)),
        }

        kwargs: dict[str, Any] = {
            "modelId": self.settings.chat_model,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        if system:
            kwargs["system"] = system

        try:
            response = await asyncio.to_thread(
                client.converse,
                **kwargs,
            )

            output_message = response.get("output", {}).get("message", {})

            content = output_message.get("content", [])

            reply_parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]

            reply = "".join(reply_parts)

            if not reply:
                raise ProviderExecutionError("Bedrock response did not contain text output.")

            usage = response.get("usage", {})

            return {
                "reply": reply,
                "usage": {
                    "tokens_in": usage.get("inputTokens", 0),
                    "tokens_out": usage.get("outputTokens", 0),
                },
                "tool_calls": [],
                "stop_reason": response.get("stopReason"),
            }

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
            logger.exception("Bedrock chat request failed.")
            self._raise_provider_error(exc)
            raise AssertionError("unreachable")

    async def stream(
        self,
        request: dict[str, Any],
    ) -> AsyncIterator[str]:
        model = request.get("model", "bedrock-chat")

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Bedrock chat model: {model}")

        client = self._require_client()

        messages, system = self._messages_from_request(request)

        kwargs: dict[str, Any] = {
            "modelId": self.settings.chat_model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": int(request.get("max_tokens", 1024)),
                "temperature": float(request.get("temperature", 0.7)),
            },
        }

        if system:
            kwargs["system"] = system

        try:
            response = await asyncio.to_thread(
                client.converse_stream,
                **kwargs,
            )

            stream = response["stream"]
            sentinel = object()
            iterator = iter(stream)

            while True:
                event = await asyncio.to_thread(
                    next,
                    iterator,
                    sentinel,
                )

                if event is sentinel:
                    break

                delta = event.get("contentBlockDelta", {}).get("delta", {}).get("text")

                if isinstance(delta, str) and delta:
                    yield delta

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
            logger.exception("Bedrock streaming request failed.")
            self._raise_provider_error(exc)

    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        model = request.get("model")

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Bedrock embedding model: {model}")

        text = request.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Bedrock embedding text must not be empty.")

        client = self._require_client()

        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self.settings.embedding_dimensions,
                "normalize": self.settings.embedding_normalize,
            }
        )

        try:
            response = await asyncio.to_thread(
                client.invoke_model,
                modelId=self.settings.embedding_model,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())

            embedding = response_body.get("embedding")

            if not isinstance(embedding, list) or not embedding:
                raise ProviderExecutionError("Bedrock embedding response did not contain a vector.")

            if not all(isinstance(value, (int, float)) for value in embedding):
                raise ProviderExecutionError("Bedrock embedding response contained invalid values.")

            return [float(value) for value in embedding]

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
            logger.exception("Bedrock embedding request failed.")
            self._raise_provider_error(exc)
            raise AssertionError("unreachable")

    async def health_check(
        self,
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "configured": self.client is not None,
            "region": self.settings.region,
            "chat_model": self.settings.chat_model,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
        }

    async def list_models(self) -> list[str]:
        return [
            *sorted(SUPPORTED_CHAT_MODELS),
            *sorted(SUPPORTED_EMBEDDING_MODELS),
        ]

    def supported_chat_models(self) -> list[str]:
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self) -> list[str]:
        return list(SUPPORTED_EMBEDDING_MODELS)

    def supported_stream_models(self) -> list[str]:
        return self.supported_chat_models()
