from typing import Any

from ai_platform.llm_gateway.providers.provider_loader import (
    load_providers,
)
from ai_platform.llm_gateway.registry.model_registry import (
    model_registry,
)


class ProviderRegistry:
    """
    Registry of instantiated LLM providers.

    Provider registration also registers the provider's capabilities
    with the ModelRegistry.
    """

    def __init__(self):
        self._registry: dict[str, Any] = {}

    def register(
        self,
        name: str,
        provider: Any,
    ) -> None:
        """Register a provider and its model capabilities."""

        self._registry[name] = provider

        model_registry.register_provider(
            name,
            provider,
        )

    def get_provider(
        self,
        name: str,
    ) -> Any:
        """Retrieve a provider by name."""

        provider = self._registry.get(
            name,
        )

        if provider is None:
            raise ValueError(f"Provider '{name}' not found in registry")

        return provider

    def remove_provider(
        self,
        name: str,
    ) -> None:
        """Remove a provider from the registry."""

        if name not in self._registry:
            raise ValueError(f"Provider '{name}' not found.")

        del self._registry[name]

        model_registry.unregister_provider(
            name,
        )

    def list_providers(
        self,
    ) -> list[str]:
        """List all registered provider names."""

        return list(self._registry.keys())

    def reload(
        self,
    ) -> None:
        """Reload provider plugins."""

        raise NotImplementedError

    def health(
        self,
    ) -> dict[str, Any]:
        """Return registry health."""

        raise NotImplementedError


# Initialize registry and register configured providers.
registry = ProviderRegistry()

load_providers(
    registry,
)
