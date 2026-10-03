import pytest

from ai_platform.llm_gateway.models.capabilities import ModelCapabilities
from ai_platform.llm_gateway.registry.model_registry import ModelRegistry


class FakeProvider:
    def supported_chat_models(self):
        return [
            "gpt-4.1",
            "gpt-4o",
        ]

    def supported_embedding_models(self):
        return [
            "openai-embedding",
        ]


class ChatOnlyProvider:
    def supported_chat_models(self):
        return [
            "gemini-chat",
        ]


def test_register_provider():

    registry = ModelRegistry()

    provider = FakeProvider()

    registry.register_provider(
        "openai",
        provider,
    )

    assert registry.provider_exists("openai")


def test_register_provider_discovers_chat_models():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    models = registry.get_models("openai")

    assert models["chat"] == [
        "gpt-4.1",
        "gpt-4o",
    ]


def test_register_provider_discovers_embedding_models():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    models = registry.get_models("openai")

    assert models["embeddings"] == [
        "openai-embedding",
    ]


def test_register_provider_without_embeddings():

    registry = ModelRegistry()

    registry.register_provider(
        "gemini",
        ChatOnlyProvider(),
    )

    models = registry.get_models("gemini")

    assert models["chat"] == [
        "gemini-chat",
    ]

    assert models["embeddings"] == []


def test_model_supported():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    assert registry.model_supported(
        "openai",
        "chat",
        "gpt-4.1",
    )

    assert registry.model_supported(
        "openai",
        "embeddings",
        "openai-embedding",
    )


def test_model_not_supported():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    assert not registry.model_supported(
        "openai",
        "chat",
        "unknown-model",
    )


def test_get_providers_for_model():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_provider(
        "another-provider",
        FakeProvider(),
    )

    providers = registry.get_providers_for_model(
        "chat",
        "gpt-4.1",
    )

    assert providers == [
        "openai",
        "another-provider",
    ]


def test_unregister_provider():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    assert registry.provider_exists("openai")

    registry.unregister_provider("openai")

    assert not registry.provider_exists("openai")


def test_list_providers():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_provider(
        "gemini",
        ChatOnlyProvider(),
    )

    providers = registry.list_providers()

    assert providers == [
        "openai",
        "gemini",
    ]


def test_list_models_for_capability():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_provider(
        "gemini",
        ChatOnlyProvider(),
    )

    result = registry.list_models("chat")

    assert result["openai"] == [
        "gpt-4.1",
        "gpt-4o",
    ]

    assert result["gemini"] == [
        "gemini-chat",
    ]


def test_list_all_models():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    result = registry.list_models()

    assert result["openai"] == [
        "gpt-4.1",
        "gpt-4o",
        "openai-embedding",
    ]


def test_unknown_provider_returns_empty_models():

    registry = ModelRegistry()

    assert registry.get_models("unknown") == {}


def test_clear():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.clear()

    assert registry.list_providers() == []


def test_register_logical_model():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_logical_model(
        logical_model="enterprise-chat",
        capability="chat",
        provider="openai",
        model="gpt-4o",
    )

    routes = registry.get_logical_model_routes(
        "enterprise-chat",
    )

    assert len(routes) == 1

    assert routes[0].logical_model == "enterprise-chat"
    assert routes[0].capability == "chat"
    assert routes[0].provider == "openai"
    assert routes[0].model == "gpt-4o"


def test_logical_model_can_have_multiple_provider_routes():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_provider(
        "another-openai",
        FakeProvider(),
    )

    registry.register_logical_model(
        logical_model="enterprise-chat",
        capability="chat",
        provider="openai",
        model="gpt-4o",
    )

    registry.register_logical_model(
        logical_model="enterprise-chat",
        capability="chat",
        provider="another-openai",
        model="gpt-4o",
    )

    routes = registry.get_logical_model_routes(
        "enterprise-chat",
    )

    assert [route.provider for route in routes] == [
        "openai",
        "another-openai",
    ]


def test_unknown_logical_model_returns_empty_routes():

    registry = ModelRegistry()

    routes = registry.get_logical_model_routes(
        "does-not-exist",
    )

    assert routes == []


def test_is_logical_model():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_logical_model(
        logical_model="enterprise-chat",
        capability="chat",
        provider="openai",
        model="gpt-4o",
    )

    assert registry.is_logical_model(
        "enterprise-chat",
    )

    assert not registry.is_logical_model(
        "does-not-exist",
    )


def test_logical_model_requires_registered_provider():

    registry = ModelRegistry()

    with pytest.raises(
        ValueError,
        match="Provider is not registered",
    ):
        registry.register_logical_model(
            logical_model="enterprise-chat",
            capability="chat",
            provider="openai",
            model="gpt-4o",
        )


def test_clear_removes_logical_model_routes():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_logical_model(
        logical_model="enterprise-chat",
        capability="chat",
        provider="openai",
        model="gpt-4o",
    )

    registry.clear()

    assert not registry.is_logical_model(
        "enterprise-chat",
    )


def test_logical_model_routes_can_be_filtered_by_capability():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_logical_model(
        logical_model="enterprise",
        capability="chat",
        provider="openai",
        model="gpt-4o",
    )

    registry.register_logical_model(
        logical_model="enterprise",
        capability="embeddings",
        provider="openai",
        model="openai-embedding",
    )

    chat_routes = registry.get_logical_model_routes(
        "enterprise",
        capability="chat",
    )

    embedding_routes = registry.get_logical_model_routes(
        "enterprise",
        capability="embeddings",
    )

    assert len(chat_routes) == 1
    assert chat_routes[0].model == "gpt-4o"

    assert len(embedding_routes) == 1
    assert embedding_routes[0].model == "openai-embedding"


def test_logical_model_requires_model_capability_support():

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    with pytest.raises(
        ValueError,
        match="Model is not supported for capability",
    ):
        registry.register_logical_model(
            logical_model="enterprise",
            capability="embeddings",
            provider="openai",
            model="gpt-4o",
        )


def test_get_model_capabilities_returns_none_for_legacy_provider():
    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    assert (
        registry.get_model_capabilities(
            "openai",
            "gpt-4.1",
        )
        is None
    )


def test_register_provider_discovers_optional_model_capabilities():
    class ProviderWithCapabilities(FakeProvider):
        def model_capabilities(self):
            return {
                "gpt-4.1": ModelCapabilities(
                    context_window_tokens=128000,
                    max_output_tokens=16384,
                    supports_tools=True,
                    supports_vision=True,
                )
            }

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        ProviderWithCapabilities(),
    )

    capabilities = registry.get_model_capabilities(
        "openai",
        "gpt-4.1",
    )

    assert capabilities is not None
    assert capabilities.context_window_tokens == 128000
    assert capabilities.max_output_tokens == 16384
    assert capabilities.supports_tools is True
    assert capabilities.supports_vision is True


def test_get_model_capabilities_returns_none_for_unknown_provider():
    registry = ModelRegistry()

    assert (
        registry.get_model_capabilities(
            "unknown-provider",
            "unknown-model",
        )
        is None
    )


def test_get_model_capabilities_returns_none_for_unknown_model():
    class ProviderWithCapabilities(FakeProvider):
        def model_capabilities(self):
            return {
                "gpt-4.1": ModelCapabilities(
                    context_window_tokens=128000,
                )
            }

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        ProviderWithCapabilities(),
    )

    assert (
        registry.get_model_capabilities(
            "openai",
            "unknown-model",
        )
        is None
    )
