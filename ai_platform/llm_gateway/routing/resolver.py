from typing import Any

from ai_platform.llm_gateway.providers.resolver import (
    ProviderResolver,
)
from ai_platform.llm_gateway.registry.model_registry import (
    ModelRegistry,
)
from ai_platform.llm_gateway.registry.model_registry import (
    model_registry as default_model_registry,
)
from ai_platform.llm_gateway.routing.policies import (
    RoutingPolicy,
)


class RoutingResolver:
    """
    Resolve a request into ordered provider implementations.

    RoutingPolicy determines the candidate provider names.

    ProviderResolver resolves those names into actual provider objects.
    """

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        routing_policy: RoutingPolicy | None = None,
        provider_resolver: ProviderResolver | None = None,
    ):
        self.model_registry = (
            model_registry if model_registry is not None else default_model_registry
        )

        self.routing_policy = routing_policy or RoutingPolicy(
            model_registry=self.model_registry,
        )

        self.provider_resolver = provider_resolver or ProviderResolver(
            model_registry=self.model_registry,
        )

    def resolve(
        self,
        capability: str,
        model: str,
        requested_provider: str | None = None,
    ) -> list[Any]:
        """
        Return ordered provider implementations.
        """

        provider_names = self.routing_policy.resolve(
            capability=capability,
            model=model,
            requested_provider=requested_provider,
        )

        return self.provider_resolver.resolve_many(
            provider_names,
        )

    def resolve_names(
        self,
        capability: str,
        model: str,
        requested_provider: str | None = None,
    ) -> list[str]:
        """
        Return ordered provider names without resolving implementations.
        """

        return self.routing_policy.resolve(
            capability=capability,
            model=model,
            requested_provider=requested_provider,
        )
