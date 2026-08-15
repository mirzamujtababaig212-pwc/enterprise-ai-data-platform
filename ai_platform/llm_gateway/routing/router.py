from collections.abc import AsyncIterator
from typing import Any

from opentelemetry import trace

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.providers.provider_factory import (
    ProviderFactory,
)
from ai_platform.llm_gateway.routing.resolver import (
    RoutingResolver,
)
from ai_platform.llm_gateway.services.capability_service import (
    capability_service,
)

tracer = trace.get_tracer(__name__)


class Router:
    """
    Gateway request router.

    Routing decisions are delegated to RoutingResolver.
    Provider execution remains the responsibility of Router.
    """

    def __init__(
        self,
        routing_resolver: RoutingResolver | None = None,
    ):
        self.routing_resolver = routing_resolver or RoutingResolver()

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        provider_name = request.get("provider")
        model = request["model"]

        if provider_name is not None:
            capability_service.validate_chat(
                provider_name,
                model,
            )

        providers = self.routing_resolver.resolve(
            capability="chat",
            model=model,
            requested_provider=provider_name,
        )

        if not providers:
            raise ProviderNotFound(f"No provider supports chat model: {model}")

        provider = providers[0]

        with tracer.start_as_current_span("provider_call"):
            response = await provider.chat(request)

        with tracer.start_as_current_span("response_parsing"):
            return response

    async def route_embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        provider_name = request.get("provider")
        model = request["model"]

        if provider_name is not None:
            capability_service.validate_embeddings(
                provider_name,
                model,
            )

        providers = self.routing_resolver.resolve(
            capability="embeddings",
            model=model,
            requested_provider=provider_name,
        )

        if not providers:
            raise ProviderNotFound(f"No provider supports embeddings model: {model}")

        provider = providers[0]

        with tracer.start_as_current_span("provider_call"):
            response = await provider.embeddings(request)

        with tracer.start_as_current_span("response_parsing"):
            return response

    async def route_stream(
        self,
        request: dict[str, Any],
    ) -> AsyncIterator[str]:
        provider_name = request.get("provider")
        model = request["model"]

        if provider_name is not None:
            capability_service.validate_stream(
                provider_name,
                model,
            )

        providers = self.routing_resolver.resolve(
            capability="stream",
            model=model,
            requested_provider=provider_name,
        )

        if not providers:
            raise ProviderNotFound(f"No provider supports streaming model: {model}")

        provider = providers[0]

        stream = provider.stream(request)

        try:
            with tracer.start_as_current_span("provider_call"):
                async for chunk in stream:
                    yield chunk
        except GeneratorExit:
            raise

    async def route_health(
        self,
    ) -> dict[str, Any]:

        health: dict[str, Any] = {}

        for provider_name in ProviderFactory.list_providers():
            provider = await self._get_provider(provider_name)

            with tracer.start_as_current_span("provider_health_check") as span:
                span.set_attribute(
                    "provider.name",
                    provider_name,
                )

                health[provider_name] = await provider.health_check()

        return health

    async def route_models(
        self,
    ) -> dict[str, list[str]]:

        models: dict[str, list[str]] = {}

        for provider_name in ProviderFactory.list_providers():
            provider = await self._get_provider(provider_name)

            with tracer.start_as_current_span("provider_call"):
                models[provider_name] = await provider.list_models()

        with tracer.start_as_current_span("response_parsing"):
            return models

    async def _get_provider(
        self,
        provider_name: str,
    ) -> Any:

        with tracer.start_as_current_span("provider_selection") as span:
            span.set_attribute(
                "provider.name",
                provider_name,
            )

            provider = ProviderFactory.get_provider(provider_name)

        if not provider:
            raise ProviderNotFound(f"Unknown provider: {provider_name}")

        return provider


router = Router()
