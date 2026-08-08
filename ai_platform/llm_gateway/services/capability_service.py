from ai_platform.llm_gateway.registry.provider_capabilities import (
    model_supported,
    provider_exists,
)
from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)


class CapabilityService:

    def provider_exists(
        self,
        provider: str,
    ) -> bool:
        return provider_exists(provider)

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

        if not provider_exists(provider):
            raise ProviderNotFound(f"Unsupported provider: {provider}")

    def validate_model(
        self,
        provider: str,
        capability: str,
        model: str,
    ) -> None:

        self.validate_provider(provider)

        if not model_supported(
            provider,
            capability,
            model,
        ):
            raise ValueError(f"Unsupported {provider} " f"{capability} model: {model}")

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
