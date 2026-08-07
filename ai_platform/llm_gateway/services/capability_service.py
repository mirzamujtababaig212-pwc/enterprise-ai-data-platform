from ai_platform.llm_gateway.registry.provider_capabilities import (
    provider_exists,
    model_supported,
)


class CapabilityService:

    def provider_exists(
        self,
        provider_name: str,
    ) -> bool:
        return provider_exists(provider_name)

    def model_supported(
        self,
        provider_name: str,
        capability: str,
        model: str,
    ) -> bool:
        return model_supported(
            provider_name,
            capability,
            model,
        )


capability_service = CapabilityService()
