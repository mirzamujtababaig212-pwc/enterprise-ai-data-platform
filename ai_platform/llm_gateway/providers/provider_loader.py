from ai_platform.llm_gateway.config.provider_settings import (
    ENABLED_PROVIDERS,
)
from ai_platform.llm_gateway.providers.anthropic_provider import (
    AnthropicProvider,
)
from ai_platform.llm_gateway.providers.azure_openai_provider import (
    AzureOpenAIProvider,
)
from ai_platform.llm_gateway.providers.bedrock_provider import (
    BedrockProvider,
)
from ai_platform.llm_gateway.providers.gemini_provider import (
    GeminiProvider,
)
from ai_platform.llm_gateway.providers.ollama_provider import (
    OllamaProvider,
)
from ai_platform.llm_gateway.providers.openai_provider import (
    OpenAIProvider,
)

PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "azure_openai": AzureOpenAIProvider,
    "bedrock": BedrockProvider,
    "ollama": OllamaProvider,
}


def load_providers(registry) -> None:
    """
    Load and register all configured providers.

    Raises:
        ValueError:
            If an unknown provider is configured.
    """

    for provider_name in ENABLED_PROVIDERS:

        if provider_name not in PROVIDER_CLASSES:
            raise ValueError(f"Unknown configured provider: {provider_name}")

        provider_class = PROVIDER_CLASSES[provider_name]

        registry.register(
            provider_name,
            provider_class(),
        )
