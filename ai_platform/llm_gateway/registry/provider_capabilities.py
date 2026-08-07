PROVIDER_CAPABILITIES = {
    "openai": {
        "chat": ["openai-gpt"],
        "embeddings": ["openai-embedding"],
        "stream": ["openai-gpt"],
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
