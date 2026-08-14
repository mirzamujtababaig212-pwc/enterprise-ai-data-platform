from unittest.mock import MagicMock

from ai_platform.llm_gateway.registry.model_registry import ModelRegistry


def test_model_registry_registers_real_provider_capabilities():

    provider = MagicMock()

    provider.supported_chat_models.return_value = [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
    ]

    provider.supported_embedding_models.return_value = [
        "openai-embedding",
    ]

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        provider,
    )

    assert registry.model_supported(
        "openai",
        "chat",
        "gpt-4.1",
    )

    assert registry.model_supported(
        "openai",
        "chat",
        "gpt-4.1-mini",
    )

    assert registry.model_supported(
        "openai",
        "embeddings",
        "openai-embedding",
    )


def test_multiple_providers_can_share_model():

    provider_a = MagicMock()

    provider_a.supported_chat_models.return_value = [
        "shared-model",
    ]

    provider_a.supported_embedding_models.return_value = []

    provider_b = MagicMock()

    provider_b.supported_chat_models.return_value = [
        "shared-model",
    ]

    provider_b.supported_embedding_models.return_value = []

    registry = ModelRegistry()

    registry.register_provider(
        "provider-a",
        provider_a,
    )

    registry.register_provider(
        "provider-b",
        provider_b,
    )

    providers = registry.get_providers_for_model(
        "chat",
        "shared-model",
    )

    assert providers == [
        "provider-a",
        "provider-b",
    ]
