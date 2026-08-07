import os

DEFAULT_PROVIDERS = [
    "openai",
    "gemini",
    "anthropic",
    "azure_openai",
    "bedrock",
    "ollama",
]


def get_enabled_providers() -> list[str]:
    """
    Read enabled providers from the environment.

    Example:

    ENABLED_PROVIDERS=openai,gemini
    """

    providers = os.getenv("ENABLED_PROVIDERS")

    if not providers:
        return DEFAULT_PROVIDERS

    enabled = [provider.strip() for provider in providers.split(",") if provider.strip()]

    return enabled
