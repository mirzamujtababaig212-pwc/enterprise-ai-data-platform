from typing import Dict, Any
from ai_platform.llm_gateway.providers.openai_provider import OpenAIProvider
from ai_platform.llm_gateway.providers.gemini_provider import GeminiProvider
from ai_platform.llm_gateway.providers.anthropic_provider import AnthropicProvider
from ai_platform.llm_gateway.registry.provider_registry import registry


class Router:
    def __init__(self):
        # Registry of available providers
        self.providers: Dict[str, Any] = {
            "openai": OpenAIProvider(),
            "gemini": GeminiProvider(),
            "anthropic": AnthropicProvider(),
        }

    def route_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide which provider to use for a chat request.
        Example strategy: choose based on 'model' field in request.
        """
        model = request.get("model", "openai")  # default to OpenAI
        provider = registry.get(model)

        if not provider:
            raise ValueError(f"Unknown provider: {model}")

        return provider.chat(request)

    def route_embeddings(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route embedding requests to the correct provider.
        """
        model = request.get("model", "openai")
        provider = self.providers.get(model)

        if not provider:
            raise ValueError(f"Unknown provider: {model}")

        return provider.embeddings(request)

    def route_health(self) -> Dict[str, Any]:
        """
        Aggregate health checks from all providers.
        """
        return {name: provider.health_check() for name, provider in self.providers.items()}
