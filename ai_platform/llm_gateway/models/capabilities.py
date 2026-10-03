from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Immutable capabilities for one provider model.

    None means the capability is not known by the runtime registry.
    """

    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None


@dataclass(frozen=True)
class ContextCapabilityEnvelope:
    """Context-window capabilities for all eligible chat routes."""

    minimum_known_context_window: int | None = None
    eligible_route_count: int = 0
    known_route_count: int = 0
    unknown_route_count: int = 0


@dataclass(frozen=True)
class ProviderCapabilities:
    """
    Immutable capability snapshot for a provider.
    """

    chat: tuple[str, ...] = ()
    embeddings: tuple[str, ...] = ()
    stream: tuple[str, ...] = ()
    model_capabilities: Mapping[str, ModelCapabilities] = MappingProxyType({})

    def supports(
        self,
        capability: str,
        model: str,
    ) -> bool:
        models = {
            "chat": self.chat,
            "embeddings": self.embeddings,
            "stream": self.stream,
        }

        return model in models.get(
            capability,
            (),
        )

    def models_for(
        self,
        capability: str,
    ) -> tuple[str, ...]:
        models = {
            "chat": self.chat,
            "embeddings": self.embeddings,
            "stream": self.stream,
        }

        return models.get(capability, ())

    def get_model_capabilities(
        self,
        model: str,
    ) -> ModelCapabilities | None:
        return self.model_capabilities.get(model)


@dataclass(frozen=True)
class RegistrySnapshot:
    """
    Immutable runtime snapshot of provider capabilities.
    """

    providers: Mapping[str, ProviderCapabilities]

    @classmethod
    def empty(cls) -> "RegistrySnapshot":
        return cls(
            providers=MappingProxyType({}),
        )

    def provider_exists(
        self,
        provider: str,
    ) -> bool:
        return provider in self.providers

    def model_supported(
        self,
        provider: str,
        capability: str,
        model: str,
    ) -> bool:
        provider_capabilities = self.providers.get(
            provider,
        )

        if provider_capabilities is None:
            return False

        return provider_capabilities.supports(
            capability,
            model,
        )
