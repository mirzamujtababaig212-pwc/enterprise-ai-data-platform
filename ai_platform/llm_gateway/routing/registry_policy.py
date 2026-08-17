"""Model-registry-backed routing policy."""

from typing import Any

from ai_platform.llm_gateway.registry.model_registry import ModelRegistry
from ai_platform.llm_gateway.routing.candidate_set import CandidateSet
from ai_platform.llm_gateway.routing.candidates import RoutingCandidate
from ai_platform.llm_gateway.routing.policy import RoutingPolicy


class RegistryRoutingPolicy(RoutingPolicy):
    """Resolve routing candidates from the model registry."""

    def __init__(
        self,
        model_registry: ModelRegistry,
    ) -> None:
        self.model_registry = model_registry

    def resolve_candidates(
        self,
        request: dict[str, Any],
    ) -> CandidateSet:
        """Resolve candidates from registry information."""

        capability = request.get("capability")
        model = request.get("model")
        requested_provider = request.get("provider")

        if not capability or not model:
            return CandidateSet()

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
