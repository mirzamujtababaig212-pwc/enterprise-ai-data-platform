import pytest

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
