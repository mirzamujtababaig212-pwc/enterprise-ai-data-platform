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
from ai_platform.llm_gateway.routing.balancer import (
    RoundRobinLoadBalancer,
)
from ai_platform.llm_gateway.routing.candidate_set import (
    CandidateSet,
)
from ai_platform.llm_gateway.routing.candidates import (
    RoutingCandidate,
)
from ai_platform.llm_gateway.routing.policy import (
    ExplicitRoutingPolicy,
    RoutingPolicy,
)


class RoutingResolver:
    """
    Resolve a request into a selected provider implementation.

    Responsibilities:

    1. Determine eligible routing candidates.
    2. Apply deterministic load balancing.
    3. Resolve the selected provider into an implementation.

    This class deliberately does not execute provider calls.
    """

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        routing_policy: RoutingPolicy | None = None,
        provider_resolver: ProviderResolver | None = None,
        load_balancer: RoundRobinLoadBalancer | None = None,
    ) -> None:
        self.model_registry = (
            model_registry if model_registry is not None else default_model_registry
        )

        self.routing_policy = (
            routing_policy if routing_policy is not None else ExplicitRoutingPolicy()
        )

        self.provider_resolver = (
            provider_resolver
            if provider_resolver is not None
            else ProviderResolver(
                model_registry=self.model_registry,
            )
        )

        self.load_balancer = (
            load_balancer if load_balancer is not None else RoundRobinLoadBalancer()
        )

    def resolve(
        self,
        capability: str,
        model: str,
        requested_provider: str | None = None,
    ) -> list[Any]:
        """Return the selected provider implementation.

        The resolver currently returns the deterministic selection only.
        Future fallback logic can use the remaining candidates.
        """

        candidate_set = self.resolve_candidates(
            capability=capability,
            model=model,
            requested_provider=requested_provider,
        )

        if not candidate_set:
            return []

        candidates = list(candidate_set)

        selected = self.load_balancer.select(
            candidates,
        )

        providers = self.provider_resolver.resolve_many(
            [selected.provider],
        )

        return providers

    def resolve_candidates(
        self,
        capability: str,
        model: str,
        requested_provider: str | None = None,
    ) -> CandidateSet:
        """Resolve eligible routing candidates."""

        providers = self.model_registry.get_providers_for_model(
            capability,
            model,
        )

        if requested_provider is not None:
            providers = [provider for provider in providers if provider == requested_provider]

        candidates = [
            RoutingCandidate(
                provider=provider,
                model=model,
            )
            for provider in providers
        ]

        return CandidateSet(candidates)

    def resolve_names(
        self,
        capability: str,
        model: str,
        requested_provider: str | None = None,
    ) -> list[str]:
        """
        Return candidate provider names without resolving implementations.
        """

        candidates = self.resolve_candidates(
            capability=capability,
            model=model,
            requested_provider=requested_provider,
        )

        return [candidate.provider for candidate in candidates]
