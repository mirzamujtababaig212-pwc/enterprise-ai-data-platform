import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from opentelemetry import trace

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.metrics.prometheus import (
    PROVIDER_ERRORS_TOTAL,
    PROVIDER_LATENCY_SECONDS,
    PROVIDER_REQUESTS_TOTAL,
)
from ai_platform.llm_gateway.reliability.failure_classifier import (
    failure_classifier,
)
from ai_platform.llm_gateway.providers.provider_factory import (
    ProviderFactory,
)
from ai_platform.llm_gateway.routing.fallback_executor import FallbackExecutor
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
        fallback_executor: FallbackExecutor | None = None,
    ):
        self.routing_resolver = routing_resolver or RoutingResolver()
        self.fallback_executor = fallback_executor or FallbackExecutor()

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        result = await self.route_chat_with_metadata(request)
        return result.response

    async def route_chat_with_metadata(
        self,
        request: dict[str, Any],
    ):
        provider_name = request.get("provider")
        model = request["model"]

        if provider_name is not None and not self.routing_resolver.is_logical_model(model):
            capability_service.validate_chat(
                provider_name,
                model,
            )

        if self.routing_resolver.is_logical_model(model):
            routes = self.routing_resolver.resolve_routes(
                capability="chat",
                model=model,
                requested_provider=provider_name,
            )

            if not routes:
                raise ProviderNotFound(f"No provider supports chat model: {model}")

            providers = [route.provider for route in routes]
            physical_models = {id(route.provider): route.model for route in routes}
        else:
            providers = self.routing_resolver.resolve(
                capability="chat",
                model=model,
                requested_provider=provider_name,
            )

            if not providers:
                raise ProviderNotFound(f"No provider supports chat model: {model}")

            physical_models = {}

        with tracer.start_as_current_span("gateway.chat") as span:
            span.set_attribute(
                "llm.model",
                model,
            )

            if provider_name:
                span.set_attribute(
                    "llm.requested_provider",
                    provider_name,
                )

            async def call_provider(provider):
                physical_model = physical_models.get(id(provider))

                provider_name_for_call = getattr(
                    provider,
                    "name",
                    getattr(
                        provider,
                        "provider_name",
                        provider.__class__.__name__,
                    ),
                )

                provider_model = physical_model or model

                with tracer.start_as_current_span("provider_call") as provider_span:
                    provider_span.set_attribute(
                        "provider.name",
                        provider_name_for_call,
                    )

                    provider_span.set_attribute(
                        "provider.model",
                        provider_model,
                    )

                    try:
                        if physical_model is None:
                            return await provider.chat(request)

                        provider_request = {
                            **request,
                            "model": physical_model,
                        }

                        return await provider.chat(provider_request)

                    except Exception as exc:
                        provider_span.record_exception(exc)
                        provider_span.set_status(trace.StatusCode.ERROR)
                        raise

            result = await self.fallback_executor.execute(
                providers,
                call_provider,
            )

            span.set_attribute(
                "llm.provider",
                result.provider_name,
            )

            span.set_attribute(
                "llm.attempt_count",
                len(result.attempts),
            )

            physical_model = model

            if self.routing_resolver.is_logical_model(model):
                for provider in providers:
                    provider_name_for_route = getattr(
                        provider,
                        "name",
                        getattr(
                            provider,
                            "provider_name",
                            provider.__class__.__name__,
                        ),
                    )

                    if provider_name_for_route == result.provider_name:
                        physical_model = physical_models[id(provider)]
                        break

            return replace(
                result,
                model_name=physical_model,
            )

    async def route_embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        result = await self.route_embeddings_with_metadata(request)
        return result.response

    async def route_embeddings_with_metadata(
        self,
        request: dict[str, Any],
    ):
        provider_name = request.get("provider")
        model = request["model"]

        if provider_name is not None and not self.routing_resolver.is_logical_model(model):
            capability_service.validate_embeddings(
                provider_name,
                model,
            )

        if self.routing_resolver.is_logical_model(model):
            routes = self.routing_resolver.resolve_routes(
                capability="embeddings",
                model=model,
                requested_provider=provider_name,
            )

            if not routes:
                raise ProviderNotFound(f"No provider supports embeddings model: {model}")

            providers = [route.provider for route in routes]
            physical_models = {id(route.provider): route.model for route in routes}
        else:
            providers = self.routing_resolver.resolve(
                capability="embeddings",
                model=model,
                requested_provider=provider_name,
            )

            if not providers:
                raise ProviderNotFound(f"No provider supports embeddings model: {model}")

            physical_models = {}

        with tracer.start_as_current_span("gateway.embeddings") as span:
            span.set_attribute(
                "llm.model",
                model,
            )

            if provider_name:
                span.set_attribute(
                    "llm.requested_provider",
                    provider_name,
                )

            async def call_provider(provider):
                physical_model = physical_models.get(id(provider))

                provider_name_for_call = getattr(
                    provider,
                    "name",
                    getattr(
                        provider,
                        "provider_name",
                        provider.__class__.__name__,
                    ),
                )

                provider_model = physical_model or model

                with tracer.start_as_current_span("provider_call") as provider_span:
                    provider_span.set_attribute(
                        "provider.name",
                        provider_name_for_call,
                    )

                    provider_span.set_attribute(
                        "provider.model",
                        provider_model,
                    )

                    try:
                        if physical_model is None:
                            return await provider.embeddings(request)

                        provider_request = {
                            **request,
                            "model": physical_model,
                        }

                        return await provider.embeddings(provider_request)

                    except Exception as exc:
                        provider_span.record_exception(exc)
                        provider_span.set_status(trace.StatusCode.ERROR)
                        raise

            result = await self.fallback_executor.execute(
                providers,
                call_provider,
            )

            span.set_attribute(
                "llm.provider",
                result.provider_name,
            )

            span.set_attribute(
                "llm.attempt_count",
                len(result.attempts),
            )

            physical_model = model

            if self.routing_resolver.is_logical_model(model):
                for provider in providers:
                    provider_name_for_route = getattr(
                        provider,
                        "name",
                        getattr(
                            provider,
                            "provider_name",
                            provider.__class__.__name__,
                        ),
                    )

                    if provider_name_for_route == result.provider_name:
                        physical_model = physical_models[id(provider)]
                        break

            return replace(
                result,
                model_name=physical_model,
            )

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

        provider_name_for_call = getattr(
            provider,
            "name",
            getattr(
                provider,
                "provider_name",
                provider.__class__.__name__,
            ),
        )

        started_at = time.perf_counter()

        PROVIDER_REQUESTS_TOTAL.labels(
            provider=provider_name_for_call,
        ).inc()

        with tracer.start_as_current_span("gateway.stream") as gateway_span:
            gateway_span.set_attribute(
                "llm.model",
                model,
            )

            if provider_name:
                gateway_span.set_attribute(
                    "llm.requested_provider",
                    provider_name,
                )

            with tracer.start_as_current_span("provider_call") as provider_span:
                provider_span.set_attribute(
                    "provider.name",
                    provider_name_for_call,
                )

                provider_span.set_attribute(
                    "provider.model",
                    model,
                )

                try:
                    stream = provider.stream(request)

                    async for chunk in stream:
                        yield chunk

                    provider_span.set_status(
                        trace.StatusCode.OK,
                    )
                    gateway_span.set_status(
                        trace.StatusCode.OK,
                    )

                except GeneratorExit:
                    raise

                except Exception as exc:
                    category = failure_classifier.classify(exc)
                    PROVIDER_ERRORS_TOTAL.labels(
                        provider=provider_name_for_call,
                        error_type=category.value,
                    ).inc()
                    provider_span.record_exception(exc)
                    provider_span.set_attribute(
                        "provider.stream.error",
                        True,
                    )
                    provider_span.set_status(
                        trace.StatusCode.ERROR,
                    )
                    gateway_span.record_exception(exc)
                    gateway_span.set_status(
                        trace.StatusCode.ERROR,
                    )
                    raise

                finally:
                    PROVIDER_LATENCY_SECONDS.labels(
                        provider=provider_name_for_call,
                    ).observe(time.perf_counter() - started_at)

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
