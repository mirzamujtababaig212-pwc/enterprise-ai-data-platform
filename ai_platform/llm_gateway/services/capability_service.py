from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)

from ai_platform.llm_gateway.registry.provider_capabilities import (
    provider_exists,
    model_supported,
)


class CapabilityService:
    """
    Service responsible for validating providers, models,
    and supported capabilities.
    """

    def provider_exists(
        self,
        provider_name: str,
    ) -> bool:
        """
        Check whether the provider exists.
        """
        return provider_exists(provider_name)

    def model_supported(
        self,
        provider_name: str,
        capability: str,
        model: str,
    ) -> bool:
        """
        Check whether a model supports the requested capability.
        """
        return model_supported(
            provider_name,
            capability,
            model,
        )

    def validate_provider(
        self,
        provider_name: str,
    ) -> None:
        """
        Validate that the provider exists.
        """
        if not self.provider_exists(provider_name):
            raise ProviderNotFound(f"Unknown provider: {provider_name}")

    def validate_chat(
        self,
        provider_name: str,
        model: str,
    ) -> None:
        """
        Validate chat capability.
        """
        self.validate_provider(provider_name)

        if not self.model_supported(
            provider_name,
            "chat",
            model,
        ):
            raise ValueError(f"Unsupported {provider_name} chat model: {model}")

    def validate_embeddings(
        self,
        provider_name: str,
        model: str,
    ) -> None:
        """
        Validate embedding capability.
        """
        self.validate_provider(provider_name)

        if not self.model_supported(
            provider_name,
            "embeddings",
            model,
        ):
            raise ValueError(f"Unsupported {provider_name} embedding model: {model}")

    def validate_stream(
        self,
        provider_name: str,
        model: str,
    ) -> None:
        """
        Validate streaming capability.
        """
        self.validate_provider(provider_name)

        if not self.model_supported(
            provider_name,
            "stream",
            model,
        ):
            raise ValueError(f"Unsupported {provider_name} stream model: {model}")


capability_service = CapabilityService()
