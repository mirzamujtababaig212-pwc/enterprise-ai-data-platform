from typing import Any
from ai_platform.llm_gateway.providers.azure_openai_provider import AzureOpenAIProvider
from ai_platform.llm_gateway.providers.bedrock_provider import BedrockProvider
from ai_platform.llm_gateway.providers.ollama_provider import OllamaProvider
from ai_platform.llm_gateway.providers.openai_provider import OpenAIProvider
from ai_platform.llm_gateway.providers.gemini_provider import GeminiProvider
from ai_platform.llm_gateway.providers.anthropic_provider import AnthropicProvider


class ProviderRegistry:
    def __init__(self):
        self._registry: dict[str, Any] = {}

    def register(self, name: str, provider: Any) -> None:
        """Register a provider under a given name."""
        self._registry[name] = provider

    def get_provider(self, name: str) -> Any:
        """Retrieve a provider by name."""
        provider = self._registry.get(name)
        if not provider:
            raise ValueError(f"Provider '{name}' not found in registry")
        return provider

    def remove_provider(self, name: str) -> None:
        """Remove a provider from the registry."""
        if name not in self._registry:
            raise ValueError(f"Provider '{name}' not found.")
        del self._registry[name]

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._registry.keys())

    def reload(self) -> None:
        """Reload provider plugins."""
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        """Return registry health."""
        raise NotImplementedError


# Initialize registry and register providers
registry = ProviderRegistry()
registry.register("openai", OpenAIProvider())
registry.register("gemini", GeminiProvider())
registry.register("anthropic", AnthropicProvider())
registry.register("bedrock", BedrockProvider())
registry.register("azure_openai", AzureOpenAIProvider())
registry.register("ollama", OllamaProvider())
