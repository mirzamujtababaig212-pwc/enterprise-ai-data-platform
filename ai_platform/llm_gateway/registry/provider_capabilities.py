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
    "gemini": {"chat": ["gemini-chat"], "embeddings": ["gemini-embedding"]},
    "anthropic": {"chat": ["anthropic-chat"]},
    "azure_openai": {"chat": ["azure-openai-chat"], "embeddings": ["azure-openai-embedding"]},
    "bedrock": {"chat": ["bedrock-chat"]},
    "ollama": {"chat": ["ollama-chat"]},
}


def provider_exists(provider):

    return provider in PROVIDER_CAPABILITIES


def model_supported(
    provider,
    capability,
    model,
):

    return model in PROVIDER_CAPABILITIES.get(provider, {}).get(capability, [])


def get_models(provider):

    return PROVIDER_CAPABILITIES.get(provider, {})
