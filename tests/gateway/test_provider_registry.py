import pytest

from unittest.mock import MagicMock

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


def test_reload_not_implemented():

    registry = ProviderRegistry()

    with pytest.raises(NotImplementedError):
        registry.reload()


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
