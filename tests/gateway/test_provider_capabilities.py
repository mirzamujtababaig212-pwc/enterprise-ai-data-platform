from ai_platform.llm_gateway.registry.provider_capabilities import (
    PROVIDER_CAPABILITIES,
    provider_exists,
    model_supported,
    get_models,
)


def test_provider_exists_true():

    assert provider_exists("openai") is True


def test_provider_exists_false():

    assert provider_exists("fake-provider") is False


def test_model_supported_chat():

    assert model_supported(
        "openai",
        "chat",
        "gpt-4o",
    )


def test_model_supported_embeddings():

    assert model_supported(
        "openai",
        "embeddings",
        "openai-embedding",
    )


def test_model_supported_stream():

    assert model_supported(
        "openai",
        "stream",
        "gpt-4o",
    )


def test_model_not_supported():

    assert not model_supported(
        "openai",
        "chat",
        "does-not-exist",
    )


def test_unknown_provider_returns_false():

    assert not model_supported(
        "unknown",
        "chat",
        "gpt-4o",
    )


def test_get_models_openai():

    models = get_models("openai")

    assert models == PROVIDER_CAPABILITIES["openai"]


def test_get_models_unknown_provider():

    assert get_models("unknown") == {}
