from typing import Any

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.registry.model_registry import (
    ModelRegistry,
)
from ai_platform.llm_gateway.registry.model_registry import (
    model_registry as default_model_registry,
)


class ProviderResolver:
    """
    Resolve provider names into registered provider implementations.

    RoutingPolicy decides which provider names are candidates.

    ProviderResolver converts those names into actual provider objects.
    """

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
    ):
        self.model_registry = model_registry or default_model_registry

    def resolve(
        self,
        provider_name: str,
    ) -> Any:
        """
        Return the registered provider implementation.
        """

        if not self.model_registry.provider_exists(
            provider_name,
        ):
            raise ProviderNotFound(f"Unsupported provider: {provider_name}")

        return self.model_registry.get_provider(
            provider_name,
        )

    def resolve_many(
        self,
        provider_names: list[str],
    ) -> list[Any]:
        """
        Resolve multiple provider names while preserving order.
        """

        return [self.resolve(provider_name) for provider_name in provider_names]
