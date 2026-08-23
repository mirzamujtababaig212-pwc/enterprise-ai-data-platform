import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from ai_platform.llm_gateway.config.openai_settings import (
    OpenAISettings,
    get_openai_settings,
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
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
}


SUPPORTED_EMBEDDING_MODELS = {
    "openai-embedding",
}

OPENAI_EMBEDDING_MODEL_MAP = {
    "openai-embedding": "text-embedding-3-small",
}


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        settings: OpenAISettings | None = None,
    ) -> None:
        self.settings = settings or get_openai_settings()

        self.default_model = self.settings.model

        self.client = client

        if self.client is None and self.settings.api_key:
            self.client = AsyncOpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout,
                max_retries=self.settings.max_retries,
            )

    @staticmethod
    def _rate_limit_message(exc: RateLimitError) -> str:
        """Extract the most useful OpenAI rate-limit/quota information."""

        body = getattr(exc, "body", None)

        if isinstance(body, dict):
            error_body = body.get("error")

            if isinstance(error_body, dict):
                code = error_body.get("code")
                message = error_body.get("message")

                if code:
                    if message:
                        return f"OpenAI error code: {code}. {message}"
                    return f"OpenAI error code: {code}."

                if message:
                    return str(message)

        return str(exc)

    @staticmethod
    def _is_quota_exceeded(exc: RateLimitError) -> bool:
        """Return True when OpenAI explicitly reports exhausted quota."""

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

    async def chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:

        prompt = request["prompt"]

        model = request.get(
            "model",
            self.default_model,
        )

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported OpenAI chat model: {model}")

        if self.client is None:
            raise ProviderAuthenticationError("OpenAI provider is not configured.")

        try:
            response = await self.client.responses.create(
                model=model,
                input=prompt,
            )
            reply = getattr(
                response,
                "output_text",
                None,
            )

            if reply is None:
                raise ValueError("OpenAI response did not contain output_text.")

            usage = getattr(response, "usage", None)

            tokens_in = (
                getattr(
                    usage,
                    "input_tokens",
                    0,
                )
                if usage
                else 0
            )

            tokens_out = (
                getattr(
                    usage,
                    "output_tokens",
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
            }

        except AuthenticationError as exc:
            logger.exception("OpenAI authentication failed.")

            raise ProviderAuthenticationError("OpenAI authentication failed.") from exc

        except APITimeoutError as exc:
            logger.exception("OpenAI request timed out.")

            raise ProviderTimeoutError("OpenAI request timed out.") from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to OpenAI.")

            raise ProviderConnectionError("Unable to connect to OpenAI.") from exc

        except RateLimitError as exc:
            message = self._rate_limit_message(exc)

            if self._is_quota_exceeded(exc):
                logger.exception(
                    "OpenAI quota exceeded: %s",
                    message,
                )

                raise ProviderQuotaExceededError(f"OpenAI quota exceeded: {message}") from exc

            logger.exception(
                "OpenAI rate limit exceeded: %s",
                message,
            )

            raise ProviderRateLimitError(f"OpenAI rate limit exceeded: {message}") from exc

        except ProviderAuthenticationError:
            raise

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Unexpected OpenAI provider error.")
            raise ProviderExecutionError("Unexpected OpenAI provider error.") from exc

    async def stream(
        self,
        request: dict[str, Any],
    ) -> AsyncIterator[str]:

        model = request.get(
            "model",
            self.default_model,
        )

        prompt = request["prompt"]

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported OpenAI chat model: {model}")

        if self.client is None:
            raise ProviderAuthenticationError("OpenAI provider is not configured.")

        try:
            stream = await self.client.responses.create(
                model=model,
                input=prompt,
                stream=True,
            )

            async for event in stream:
                event_type = getattr(
                    event,
                    "type",
                    None,
                )

                if event_type == "response.output_text.delta":
                    delta = getattr(
                        event,
                        "delta",
                        None,
                    )

                    if delta:
                        yield delta

                elif event_type == "response.completed":
                    logger.debug("OpenAI streaming response completed.")

        except AuthenticationError as exc:
            logger.exception("OpenAI streaming authentication failed.")

            raise ProviderAuthenticationError("OpenAI authentication failed.") from exc

        except APITimeoutError as exc:
            logger.exception("OpenAI streaming request timed out.")

            raise ProviderTimeoutError("OpenAI request timed out.") from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to OpenAI during streaming.")

            raise ProviderConnectionError("Unable to connect to OpenAI.") from exc

        except RateLimitError as exc:
            message = self._rate_limit_message(exc)

            if self._is_quota_exceeded(exc):
                logger.exception(
                    "OpenAI streaming quota exceeded: %s",
                    message,
                )

                raise ProviderQuotaExceededError(f"OpenAI quota exceeded: {message}") from exc

            logger.exception(
                "OpenAI streaming rate limit exceeded: %s",
                message,
            )

            raise ProviderRateLimitError(f"OpenAI rate limit exceeded: {message}") from exc

        except Exception as exc:
            logger.exception("Unexpected OpenAI streaming error.")
            raise ProviderConnectionError("Unexpected OpenAI streaming error.") from exc

    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:

        model = request["model"]
        text = request.get("text")

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported OpenAI embedding model: {model}")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input text must not be empty.")

        if self.client is None:
            raise ProviderAuthenticationError("OpenAI provider is not configured.")

        openai_model = OPENAI_EMBEDDING_MODEL_MAP[model]

        try:
            response = await self.client.embeddings.create(
                model=openai_model,
                input=text,
            )

            if not response.data:
                raise ValueError("OpenAI embedding response contained no data.")

            embedding = response.data[0].embedding

            if not embedding:
                raise ValueError("OpenAI embedding response contained an empty vector.")

            return embedding

        except AuthenticationError as exc:
            logger.exception("OpenAI embedding authentication failed.")

            raise ProviderAuthenticationError("OpenAI authentication failed.") from exc

        except RateLimitError as exc:
            message = self._rate_limit_message(exc)

            if self._is_quota_exceeded(exc):
                logger.exception(
                    "OpenAI embedding quota exceeded: %s",
                    message,
                )

                raise ProviderQuotaExceededError(f"OpenAI quota exceeded: {message}") from exc

            logger.exception(
                "OpenAI embedding rate limit exceeded: %s",
                message,
            )

            raise ProviderRateLimitError(f"OpenAI rate limit exceeded: {message}") from exc

        except APITimeoutError as exc:
            logger.exception("OpenAI embedding request timed out.")

            raise ProviderTimeoutError("OpenAI request timed out.") from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to OpenAI embedding API.")

            raise ProviderConnectionError("Unable to connect to OpenAI.") from exc

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Unexpected OpenAI embedding error.")

            raise ProviderConnectionError("Unexpected OpenAI embedding error.") from exc

    async def health_check(
        self,
    ) -> dict[str, Any]:

        return {
            "status": "ok",
            "configured": bool(self.settings.api_key),
            "base_url": self.settings.base_url,
            "default_model": self.default_model,
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

    def supported_stream_models(self) -> list[str]:
        return sorted(SUPPORTED_CHAT_MODELS)
