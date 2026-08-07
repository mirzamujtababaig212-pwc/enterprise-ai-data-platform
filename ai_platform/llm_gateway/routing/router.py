from typing import Any

from opentelemetry import trace

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.registry.provider_registry import registry

from ai_platform.llm_gateway.registry.provider_capabilities import (
    provider_exists,
    model_supported,
)


tracer = trace.get_tracer(__name__)


class Router:
    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Decide which provider to use for a chat request.
        """
        provider_name = request.get("provider", "openai")
        model = request["model"]
        if not provider_exists(provider_name):
            raise ProviderNotFound(f"Unknown provider: {provider_name}")
        if not model_supported(
            provider_name,
            "chat",
            model,
        ):
            raise ValueError(f"Unsupported {provider_name} chat model: {model}")

        with tracer.start_as_current_span("provider_selection") as span:
            span.set_attribute(
                "provider.name",
                provider_name,
            )

            provider = registry.get_provider(provider_name)

        if not provider:
            raise ProviderNotFound(f"Unknown provider: {provider_name}")

        with tracer.start_as_current_span("provider_call"):
            response = await provider.chat(request)

        with tracer.start_as_current_span("response_parsing"):
            return response

    async def route_embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        """
        Route embedding requests to the correct provider.
        """
        provider_name = request.get("provider", "openai")
        model = request["model"]
        if not provider_exists(provider_name):
            raise ProviderNotFound(f"Unknown provider: {provider_name}")
        if not model_supported(
            provider_name,
            "embeddings",
            model,
        ):
            raise ValueError(f"Unsupported {provider_name} chat model: {model}")
        with tracer.start_as_current_span("provider_selection") as span:
            span.set_attribute(
                "provider.name",
                provider_name,
            )

            provider = registry.get_provider(provider_name)

        if not provider:
            raise ProviderNotFound(f"Unknown provider: {provider_name}")

        with tracer.start_as_current_span("provider_call"):
            response = await provider.embeddings(request)

        with tracer.start_as_current_span("response_parsing"):
            return response

    async def route_health(self) -> dict[str, Any]:
        """
        Aggregate health checks from all providers.
        """
        health: dict[str, Any] = {}

        for provider_name in registry.list_providers():
            provider = registry.get_provider(provider_name)

            if not provider:
                raise ProviderNotFound(f"Unknown provider: {provider_name}")

            with tracer.start_as_current_span("provider_health_check") as span:
                span.set_attribute(
                    "provider.name",
                    provider_name,
                )

                health[provider_name] = await provider.health_check()

        return health

    async def route_stream(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Route streaming chat requests to the correct provider.
        """
        provider_name = request.get("provider", "openai")
        model = request["model"]
        if not provider_exists(provider_name):
            raise ProviderNotFound(f"Unknown provider: {provider_name}")
        if not model_supported(
            provider_name,
            "stream",
            model,
        ):
            raise ValueError(f"Unsupported {provider_name} chat model: {model}")
        with tracer.start_as_current_span("provider_selection") as span:
            span.set_attribute(
                "provider.name",
                provider_name,
            )

            provider = registry.get_provider(provider_name)

        if not provider:
            raise ProviderNotFound(f"Unknown provider: {provider_name}")

        with tracer.start_as_current_span("provider_call"):
            response = await provider.stream(request)

        with tracer.start_as_current_span("response_parsing"):
            return response

    async def route_models(
        self,
    ) -> dict[str, list[str]]:
        """
        Return all available models from every registered provider.
        """
        models: dict[str, list[str]] = {}

        for provider_name in registry.list_providers():

            with tracer.start_as_current_span("provider_selection") as span:
                span.set_attribute(
                    "provider.name",
                    provider_name,
                )

                provider = registry.get_provider(provider_name)

            if not provider:
                raise ProviderNotFound(f"Unknown provider: {provider_name}")

            with tracer.start_as_current_span("provider_call"):
                models[provider_name] = await provider.list_models()

        with tracer.start_as_current_span("response_parsing"):
            return models


router = Router()
