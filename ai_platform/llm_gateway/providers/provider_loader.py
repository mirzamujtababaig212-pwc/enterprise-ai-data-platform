from ai_platform.llm_gateway.config.provider_settings import ENABLED_PROVIDERS
from ai_platform.llm_gateway.providers.openai_provider import OpenAIProvider
from ai_platform.llm_gateway.providers.gemini_provider import GeminiProvider
from ai_platform.llm_gateway.providers.anthropic_provider import AnthropicProvider
from ai_platform.llm_gateway.providers.azure_openai_provider import AzureOpenAIProvider
from ai_platform.llm_gateway.providers.bedrock_provider import BedrockProvider
from ai_platform.llm_gateway.providers.ollama_provider import OllamaProvider

PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "azure_openai": AzureOpenAIProvider,
    "bedrock": BedrockProvider,
    "ollama": OllamaProvider,
}


def load_providers(registry):
    for provider_name in ENABLED_PROVIDERS:
        provider_class = PROVIDER_CLASSES[provider_name]
        registry.register(
            provider_name,
            provider_class(),
        )
