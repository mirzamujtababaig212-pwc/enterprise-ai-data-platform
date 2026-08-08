from unittest.mock import patch

from ai_platform.llm_gateway.providers.provider_factory import (
    ProviderFactory,
)


def test_get_provider():

    fake_provider = object()

    with patch(
        "ai_platform.llm_gateway.providers.provider_factory.registry.get_provider",
        return_value=fake_provider,
    ) as mock_get:

        provider = ProviderFactory.get_provider("openai")

        assert provider is fake_provider

        mock_get.assert_called_once_with("openai")


def test_list_providers():

    providers = [
        "openai",
        "gemini",
        "anthropic",
    ]

    with patch(
        "ai_platform.llm_gateway.providers.provider_factory.registry.list_providers",
        return_value=providers,
    ) as mock_list:

        result = ProviderFactory.list_providers()

        assert result == providers

        mock_list.assert_called_once()
