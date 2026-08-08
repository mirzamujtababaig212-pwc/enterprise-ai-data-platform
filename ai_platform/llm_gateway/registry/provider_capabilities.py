PROVIDER_CAPABILITIES = {
    "openai": {
        "chat": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "o4-mini",
        ],
        "embeddings": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "openai-embedding",
        ],
        "stream": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "o4-mini",
        ],
    },
    "gemini": {
        "chat": ["gemini-chat"],
        "embeddings": ["gemini-embedding"],
        "stream": ["gemini-chat"],
    },
    "anthropic": {
        "chat": ["anthropic-chat"],
        "embeddings": ["anthropic-embedding"],
        "stream": ["anthropic-chat"],
    },
    "azure_openai": {
        "chat": ["azure-openai-chat"],
        "embeddings": ["azure-openai-embedding"],
        "stream": ["azure-openai-chat"],
    },
    "bedrock": {
        "chat": ["bedrock-chat"],
        "embeddings": ["bedrock-embedding"],
        "stream": ["bedrock-chat"],
    },
    "ollama": {
        "chat": ["ollama-chat"],
        "embeddings": ["ollama-embedding"],
        "stream": ["ollama-chat"],
    },
}


def provider_supported(provider: str) -> bool:
    return provider in PROVIDER_CAPABILITIES


def provider_exists(provider: str) -> bool:
    return provider in PROVIDER_CAPABILITIES


def model_supported(
    provider: str,
    capability: str,
    model: str,
) -> bool:
    return model in (PROVIDER_CAPABILITIES.get(provider, {}).get(capability, []))


def get_provider_capabilities(
    provider: str,
) -> dict:
    return PROVIDER_CAPABILITIES.get(
        provider,
        {},
    )


def get_models(provider: str) -> dict:
    return PROVIDER_CAPABILITIES.get(provider, {})
