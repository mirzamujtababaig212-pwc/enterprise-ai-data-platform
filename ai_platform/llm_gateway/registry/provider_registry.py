from typing import Dict, Any
from ai_platform.llm_gateway.providers.openai_provider import OpenAIProvider
from ai_platform.llm_gateway.providers.gemini_provider import GeminiProvider
from ai_platform.llm_gateway.providers.anthropic_provider import AnthropicProvider


class ProviderRegistry:
    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, provider: Any) -> None:
        """Register a provider under a given name."""
        self._registry[name] = provider

    def get(self, name: str) -> Any:
        """Retrieve a provider by name."""
        provider = self._registry.get(name)
        if not provider:
            raise ValueError(f"Provider '{name}' not found in registry")
        return provider

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._registry.keys())


# Initialize registry and register providers
registry = ProviderRegistry()
registry.register("openai", OpenAIProvider())
registry.register("gemini", GeminiProvider())
registry.register("anthropic", AnthropicProvider())
