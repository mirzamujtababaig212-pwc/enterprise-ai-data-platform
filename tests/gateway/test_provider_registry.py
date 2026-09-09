from unittest.mock import MagicMock

import pytest

from ai_platform.llm_gateway.registry.model_registry import (
    model_registry,
)
from ai_platform.llm_gateway.registry.provider_registry import (
    ProviderRegistry,
)

###############################################################################
# register()
###############################################################################


def test_register_provider():

    registry = ProviderRegistry()

    provider = object()

    registry.register(
        "openai",
        provider,
    )

    assert registry.get_provider("openai") is provider


###############################################################################
# get_provider()
###############################################################################


def test_get_provider_success():

    registry = ProviderRegistry()

    provider = object()

    registry.register(
        "openai",
        provider,
    )

    assert registry.get_provider("openai") is provider


def test_get_provider_not_found():

    registry = ProviderRegistry()

    with pytest.raises(ValueError):
        registry.get_provider("missing")


###############################################################################
# remove_provider()
###############################################################################


def test_remove_provider_success():

    registry = ProviderRegistry()

    provider = object()

    registry.register(
        "openai",
        provider,
    )

    registry.remove_provider("openai")

    assert registry.list_providers() == []


def test_remove_provider_not_found():

    registry = ProviderRegistry()

    with pytest.raises(ValueError):
        registry.remove_provider("missing")


###############################################################################
# list_providers()
###############################################################################


def test_list_providers():

    registry = ProviderRegistry()

    registry.register(
        "openai",
        object(),
    )

    registry.register(
        "gemini",
        object(),
    )

    providers = registry.list_providers()

    assert "openai" in providers
    assert "gemini" in providers
    assert len(providers) == 2


###############################################################################
# reload()
###############################################################################


def test_reload_reloads_configured_providers():

    registry = ProviderRegistry()

    registry.register(
        "temporary-provider",
        object(),
    )

    assert registry.list_providers() == [
        "temporary-provider",
    ]

    registry.reload()

    providers = registry.list_providers()

    assert "temporary-provider" not in providers
    assert "openai" in providers
    assert "gemini" in providers
    assert "anthropic" in providers
    assert "azure_openai" in providers
    assert "bedrock" in providers
    assert "ollama" in providers


def test_reload_bootstraps_default_logical_models(monkeypatch) -> None:
    configured_providers = {
        "openai": "gpt-4.1-mini",
        "gemini": "gemini-chat",
        "anthropic": "anthropic-chat",
        "azure_openai": "azure-openai-chat",
        "ollama": "ollama-chat",
    }

    provider_classes = {}

    for provider_name, model in configured_providers.items():
        provider = MagicMock()
        provider.is_configured = True
        provider.supported_chat_models.return_value = [model]
        provider.supported_embedding_models.return_value = []
        provider.supported_stream_models.return_value = []

        provider_classes[provider_name] = lambda provider=provider: provider

    monkeypatch.setattr(
        "ai_platform.llm_gateway.providers.provider_loader.PROVIDER_CLASSES",
        provider_classes,
    )
    monkeypatch.setattr(
        "ai_platform.llm_gateway.providers.provider_loader.get_enabled_providers",
        lambda: list(configured_providers),
    )

    registry = ProviderRegistry()
    registry.reload()

    routes = model_registry.get_logical_model_routes(
        "enterprise-chat",
        "chat",
    )

    route_map = {route.provider: route.model for route in routes}

    assert route_map == configured_providers


def test_reload_rebuilds_logical_model_routes() -> None:
    registry = ProviderRegistry()

    registry.reload()

    first_routes = model_registry.get_logical_model_routes(
        "enterprise-chat",
        "chat",
    )

    registry.reload()

    second_routes = model_registry.get_logical_model_routes(
        "enterprise-chat",
        "chat",
    )

    assert first_routes == second_routes


###############################################################################
# health()
###############################################################################


def test_health_not_implemented():

    registry = ProviderRegistry()

    with pytest.raises(NotImplementedError):
        registry.health()


def test_register_provider_updates_model_registry():

    model_registry.clear()

    registry = ProviderRegistry()

    provider = MagicMock()

    provider.supported_chat_models.return_value = [
        "test-chat",
    ]

    provider.supported_embedding_models.return_value = [
        "test-embedding",
    ]

    registry.register(
        "test-provider",
        provider,
    )

    assert model_registry.provider_exists(
        "test-provider",
    )

    assert model_registry.model_supported(
        "test-provider",
        "chat",
        "test-chat",
    )

    assert model_registry.model_supported(
        "test-provider",
        "embeddings",
        "test-embedding",
    )

    model_registry.clear()


def test_remove_provider_updates_model_registry():

    model_registry.clear()

    registry = ProviderRegistry()

    provider = MagicMock()

    provider.supported_chat_models.return_value = [
        "test-chat",
    ]

    provider.supported_embedding_models.return_value = []

    registry.register(
        "test-provider",
        provider,
    )

    assert model_registry.provider_exists(
        "test-provider",
    )

    registry.remove_provider(
        "test-provider",
    )

    assert not model_registry.provider_exists(
        "test-provider",
    )

    model_registry.clear()
