"""Bootstrap default logical-model routing definitions."""

from ai_platform.llm_gateway.registry.model_registry import ModelRegistry

DEFAULT_LOGICAL_MODEL_ROUTES = (
    (
        "enterprise-chat",
        "chat",
        "openai",
        "gpt-4.1-mini",
    ),
    (
        "enterprise-chat",
        "chat",
        "gemini",
        "gemini-chat",
    ),
    (
        "enterprise-chat",
        "chat",
        "anthropic",
        "anthropic-chat",
    ),
    (
        "enterprise-chat",
        "chat",
        "azure_openai",
        "azure-openai-chat",
    ),
    (
        "enterprise-chat",
        "chat",
        "ollama",
        "ollama-chat",
    ),
)


def register_default_logical_models(
    model_registry: ModelRegistry,
) -> None:
    """Register the platform's default logical-model routes."""

    for (
        logical_model,
        capability,
        provider,
        model,
    ) in DEFAULT_LOGICAL_MODEL_ROUTES:
        if not model_registry.provider_exists(provider):
            continue

        provider_instance = model_registry.get_provider(provider)

        if not provider_instance.is_configured:
            continue

        if not model_registry.model_supported(
            provider,
            capability,
            model,
        ):
            continue

        model_registry.register_logical_model(
            logical_model=logical_model,
            capability=capability,
            provider=provider,
            model=model,
        )
