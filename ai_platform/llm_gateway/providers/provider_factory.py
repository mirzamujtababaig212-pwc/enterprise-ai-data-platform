from typing import Any

from ai_platform.llm_gateway.registry.provider_registry import registry


class ProviderFactory:
    @staticmethod
    def get_provider(provider_name: str) -> Any:
        return registry.get_provider(provider_name)

    @staticmethod
    def list_providers() -> list[str]:
        return registry.list_providers()
