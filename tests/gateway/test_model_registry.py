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
