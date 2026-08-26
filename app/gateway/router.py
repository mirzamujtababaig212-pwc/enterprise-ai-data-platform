from __future__ import annotations

from app.providers.base import LLMProvider, LLMRequest, LLMResponse


class ProviderRouter:

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        default_provider: str,
    ) -> None:
        self.providers = providers
        self.default_provider = default_provider

    def get_provider(
        self,
        provider_name: str | None = None,
    ) -> LLMProvider:

        name = (provider_name or self.default_provider).lower()

        provider = self.providers.get(name)

        if provider is None:
            available = ", ".join(sorted(self.providers))

            raise ValueError(f"Unknown provider '{name}'. " f"Available providers: {available}")

        return provider

    async def generate(
        self,
        request: LLMRequest,
        provider_name: str | None = None,
    ) -> LLMResponse:

        provider = self.get_provider(provider_name)

        return await provider.generate(request)
