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
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
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
    def __init__(self):
        self.default_model = OPENAI_MODEL

        self.client = None

        if OPENAI_API_KEY:
            self.client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
            )

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

        try:
            if self.client is None:
                raise ProviderAuthenticationError("OpenAI provider is not configured.")

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

            tokens_in = 0
            tokens_out = 0

            if usage is not None:
                tokens_in = getattr(
                    usage,
                    "input_tokens",
                    0,
                )

                tokens_out = getattr(
                    usage,
                    "output_tokens",
                    0,
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

            raise ProviderAuthenticationError(str(exc)) from exc

        except RateLimitError as exc:
            logger.exception("OpenAI rate limit exceeded.")

            raise ProviderRateLimitError(
                "OpenAI API quota exceeded. Please verify your billing or try again later."
            ) from exc

        except APITimeoutError as exc:
            logger.exception("OpenAI request timed out.")

            raise ProviderTimeoutError(str(exc)) from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to OpenAI.")

            raise ProviderConnectionError(str(exc)) from exc

        except ProviderAuthenticationError:
            raise

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Unexpected OpenAI provider error.")
            raise ProviderConnectionError(str(exc)) from exc

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

            raise ProviderAuthenticationError(str(exc)) from exc

        except RateLimitError as exc:
            logger.exception("OpenAI streaming rate limit exceeded.")

            raise ProviderRateLimitError(
                "OpenAI API quota exceeded. Please verify your billing or try again later."
            ) from exc

        except APITimeoutError as exc:
            logger.exception("OpenAI streaming request timed out.")

            raise ProviderTimeoutError(str(exc)) from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to OpenAI during streaming.")

            raise ProviderConnectionError(str(exc)) from exc

        except Exception:
            logger.exception("Unexpected OpenAI streaming error.")

            raise

    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:

        model = request["model"]
        text = request.get("text")

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported OpenAI embedding model: {model}")

        if not text:
            return [
                0.1,
                0.2,
                0.3,
            ]

        if self.client is None:
            raise ProviderAuthenticationError("OpenAI provider is not configured.")

        openai_model = OPENAI_EMBEDDING_MODEL_MAP[model]

        logger.info(
            "Calling OpenAI embedding model",
            extra={
                "gateway_model": model,
                "provider_model": openai_model,
            },
        )

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

            raise ProviderAuthenticationError(str(exc)) from exc

        except RateLimitError as exc:
            logger.exception("OpenAI embedding rate limit exceeded.")

            raise ProviderRateLimitError(
                "OpenAI API quota exceeded. Please verify your billing or try again later."
            ) from exc

        except APITimeoutError as exc:
            logger.exception("OpenAI embedding request timed out.")

            raise ProviderTimeoutError(str(exc)) from exc

        except APIConnectionError as exc:
            logger.exception("Unable to connect to OpenAI embedding API.")

            raise ProviderConnectionError(str(exc)) from exc

        except ProviderAuthenticationError:
            raise

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Unexpected OpenAI embedding error.")

            raise ProviderConnectionError(str(exc)) from exc

    async def health_check(
        self,
    ) -> dict[str, Any]:

        return {
            "status": "ok",
            "configured": bool(OPENAI_API_KEY),
            "base_url": OPENAI_BASE_URL,
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

        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(
        self,
    ) -> list[str]:

        return list(SUPPORTED_EMBEDDING_MODELS)

    def supported_stream_models(self) -> list[str]:
        return list(SUPPORTED_CHAT_MODELS)
