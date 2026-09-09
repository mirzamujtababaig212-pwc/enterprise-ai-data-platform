from unittest.mock import MagicMock

from ai_platform.llm_gateway.registry.logical_model_bootstrap import (
    register_default_logical_models,
)
from ai_platform.llm_gateway.registry.model_registry import ModelRegistry


def test_register_default_logical_models() -> None:
    registry = ModelRegistry()

    providers = {
        "openai": ["gpt-4.1-mini"],
        "gemini": ["gemini-chat"],
        "anthropic": ["anthropic-chat"],
        "azure_openai": ["azure-openai-chat"],
        "ollama": ["ollama-chat"],
    }

    for provider_name, models in providers.items():
        provider = MagicMock()
        provider.supported_chat_models.return_value = models
        provider.supported_embedding_models.return_value = []
        provider.supported_stream_models.return_value = []

        registry.register_provider(
            provider_name,
            provider,
        )

    register_default_logical_models(
        registry,
    )

    routes = registry.get_logical_model_routes(
        "enterprise-chat",
        "chat",
    )

    assert [(route.provider, route.model) for route in routes] == [
        ("openai", "gpt-4.1-mini"),
        ("gemini", "gemini-chat"),
        ("anthropic", "anthropic-chat"),
        ("azure_openai", "azure-openai-chat"),
        ("ollama", "ollama-chat"),
    ]


def test_register_default_logical_models_skips_unconfigured_providers() -> None:
    registry = ModelRegistry()

    provider = MagicMock()
    provider.supported_chat_models.return_value = [
        "gpt-4.1-mini",
    ]
    provider.supported_embedding_models.return_value = []
    provider.supported_stream_models.return_value = []

    registry.register_provider(
        "openai",
        provider,
    )

    register_default_logical_models(
        registry,
    )

    routes = registry.get_logical_model_routes(
        "enterprise-chat",
        "chat",
    )

    assert [(route.provider, route.model) for route in routes] == [
        ("openai", "gpt-4.1-mini"),
    ]


def test_register_default_logical_models_is_idempotent() -> None:
    registry = ModelRegistry()

    provider = MagicMock()
    provider.supported_chat_models.return_value = [
        "gpt-4.1-mini",
    ]
    provider.supported_embedding_models.return_value = []
    provider.supported_stream_models.return_value = []

    registry.register_provider(
        "openai",
        provider,
    )

    register_default_logical_models(registry)
    register_default_logical_models(registry)

    routes = registry.get_logical_model_routes(
        "enterprise-chat",
        "chat",
    )

    assert len(routes) == 1
