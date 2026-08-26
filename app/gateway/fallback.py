from __future__ import annotations

import logging

from app.gateway.retry import with_retry
from app.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class ProviderFallback:

    def __init__(
        self,
        providers: dict[str, LLMProvider],
    ) -> None:
        self.providers = providers

    async def generate(
        self,
        request: LLMRequest,
        provider_order: list[str],
    ) -> LLMResponse:

        errors: list[str] = []

        for provider_name in provider_order:

            provider = self.providers.get(provider_name)

            if provider is None:
                errors.append(f"{provider_name}: provider not configured")
                continue

            try:
                logger.info(
                    "Attempting provider=%s",
                    provider_name,
                )

                return await with_retry(
                    lambda: provider.generate(request),
                    max_attempts=3,
                )

            except Exception as exc:
                logger.exception(
                    "Provider failed provider=%s",
                    provider_name,
                )

                errors.append(f"{provider_name}: {exc}")

        raise RuntimeError("All providers failed: " + " | ".join(errors))
