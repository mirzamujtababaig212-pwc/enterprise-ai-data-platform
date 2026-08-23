from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.registry.model_registry import (
    ModelRegistry,
)
from ai_platform.llm_gateway.registry.model_registry import (
    model_registry as default_model_registry,
)

SUPPORTED_CAPABILITIES = {
    "chat",
    "embeddings",
    "stream",
}


class RoutingPolicy:
    """
    Resolve an ordered set of provider candidates for a request.

    Provider/model capability information comes from ModelRegistry.
    Provider names are never hardcoded here.
    """

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
    ):
        self.model_registry = (
            model_registry if model_registry is not None else default_model_registry
        )

    def resolve(
        self,
        capability: str,
        model: str,
        requested_provider: str | None = None,
    ) -> list[str]:

        if capability not in SUPPORTED_CAPABILITIES:
            return []

        if requested_provider is not None:
            return [
                self._resolve_explicit_provider(
                    requested_provider,
                    capability,
                    model,
                )
            ]

        return self.model_registry.get_providers_for_model(
            capability,
            model,
        )

    def _resolve_explicit_provider(
        self,
        provider: str,
        capability: str,
        model: str,
    ) -> str:
        if not self.model_registry.provider_exists(
            provider,
        ):
            raise ProviderNotFound(f"Unsupported provider: {provider}")

        if not self.model_registry.model_supported(
            provider,
            capability,
            model,
        ):
            raise ValueError(f"Unsupported {provider} {capability} model: {model}")

        return provider
