from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.registry.model_registry import (
    model_registry,
)
from ai_platform.llm_gateway.registry.provider_capabilities import (
    model_supported as static_model_supported,
)
from ai_platform.llm_gateway.registry.provider_capabilities import (
    provider_exists as static_provider_exists,
)


def provider_exists(
    provider: str,
) -> bool:
    """
    Compatibility wrapper for provider existence checks.

    Runtime ModelRegistry is preferred when the provider has been
    registered. The static capability catalog is retained as a
    compatibility fallback.
    """

    if model_registry.provider_exists(
        provider,
    ):
        return True

    return static_provider_exists(
        provider,
    )


def model_supported(
    provider: str,
    capability: str,
    model: str,
) -> bool:
    """
    Compatibility wrapper for model capability checks.

    Runtime provider capabilities are the primary source of truth.
    The static capability catalog remains a compatibility fallback
    for providers that have not yet been instantiated.
    """

    if model_registry.provider_exists(
        provider,
    ):
        return model_registry.model_supported(
            provider,
            capability,
            model,
        )

    return static_model_supported(
        provider,
        capability,
        model,
    )


class CapabilityService:
    """
    Validate providers and model capabilities.

    Runtime provider capabilities are preferred whenever providers
    are registered in ModelRegistry.
    """

    def provider_exists(
        self,
        provider: str,
    ) -> bool:
        return provider_exists(
            provider,
        )

    def model_supported(
        self,
        provider: str,
        capability: str,
        model: str,
    ) -> bool:
        return model_supported(
            provider,
            capability,
            model,
        )

    def validate_provider(
        self,
        provider: str,
    ) -> None:
        if not self.provider_exists(
            provider,
        ):
            raise ProviderNotFound(f"Unsupported provider: {provider}")

    def validate_model(
        self,
        provider: str,
        capability: str,
        model: str,
    ) -> None:
        self.validate_provider(
            provider,
        )

        if not self.model_supported(
            provider,
            capability,
            model,
        ):
            raise ValueError(f"Unsupported " f"{provider} " f"{capability} " f"model: " f"{model}")

    def validate_chat(
        self,
        provider: str,
        model: str,
    ) -> None:
        self.validate_model(
            provider,
            "chat",
            model,
        )

    def validate_embeddings(
        self,
        provider: str,
        model: str,
    ) -> None:
        self.validate_model(
            provider,
            "embeddings",
            model,
        )

    def validate_stream(
        self,
        provider: str,
        model: str,
    ) -> None:
        self.validate_model(
            provider,
            "stream",
            model,
        )


capability_service = CapabilityService()
