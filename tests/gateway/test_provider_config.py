from ai_platform.llm_gateway.config.provider_config import ProviderConfig


def test_provider_config_attributes():

    assert hasattr(ProviderConfig, "OPENAI_KEY")
    assert hasattr(ProviderConfig, "GEMINI_KEY")
    assert hasattr(ProviderConfig, "ANTHROPIC_KEY")
    assert hasattr(ProviderConfig, "OLLAMA_URL")
    assert hasattr(ProviderConfig, "AZURE_ENDPOINT")
    assert hasattr(ProviderConfig, "BEDROCK_REGION")


def test_provider_config_defaults():

    assert ProviderConfig.OPENAI_KEY is Ellipsis
    assert ProviderConfig.GEMINI_KEY is Ellipsis
    assert ProviderConfig.ANTHROPIC_KEY is Ellipsis
    assert ProviderConfig.OLLAMA_URL is Ellipsis
    assert ProviderConfig.AZURE_ENDPOINT is Ellipsis
    assert ProviderConfig.BEDROCK_REGION is Ellipsis
