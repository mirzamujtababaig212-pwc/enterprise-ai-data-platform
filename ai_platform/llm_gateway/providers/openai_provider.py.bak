from typing import Any
import logging

from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)

from ai_platform.llm_gateway.config.openai_settings import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from ai_platform.llm_gateway.providers.base_provider import BaseProvider
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
)

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


class OpenAIProvider(BaseProvider):
    def __init__(self):

        self.default_model = OPENAI_MODEL

        self.client = None
        if OPENAI_API_KEY:
            self.client = OpenAI(
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

        try:

            if self.client is None:
                raise ProviderAuthenticationError("OpenAI provider is not configured.")

            response = self.client.responses.create(
                model=model,
                input=prompt,
            )

            usage = response.usage

            tokens_in = 0
            tokens_out = 0

            if usage:
                tokens_in = usage.input_tokens
                tokens_out = usage.output_tokens

            return {
                "reply": response.output_text,
                "usage": {
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
            }

        except AuthenticationError as e:

            logger.exception("OpenAI authentication failed.")

            raise ProviderAuthenticationError(str(e)) from e

        except RateLimitError as e:

            logger.exception("OpenAI rate limit exceeded.")

            raise ProviderRateLimitError(
                "OpenAI API quota exceeded. Please verify your billing or try again later."
            ) from e

        except APITimeoutError as e:

            logger.exception("OpenAI request timed out.")

            raise ProviderTimeoutError(str(e)) from e

        except APIConnectionError as e:

            logger.exception("Unable to connect to OpenAI.")

            raise ProviderConnectionError(str(e)) from e

        except Exception:

            logger.exception("Unexpected OpenAI provider error.")

            raise

    async def stream(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"stream": ["openai-chunk1", "openai-chunk2"]}

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        model = request["model"]

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported OpenAI embedding model: {model}")

        return [0.1, 0.2, 0.3]

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "configured": bool(OPENAI_API_KEY),
            "base_url": OPENAI_BASE_URL,
            "default_model": self.default_model,
        }

    async def list_models(self) -> list[str]:
        return [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "o4-mini",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "openai-embedding",
        ]

    def supported_chat_models(self) -> list[str]:
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self) -> list[str]:
        return list(SUPPORTED_EMBEDDING_MODELS)
