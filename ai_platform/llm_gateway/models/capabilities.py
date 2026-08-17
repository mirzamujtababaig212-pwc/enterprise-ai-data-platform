from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ProviderCapabilities:
    """
    Immutable capability snapshot for a provider.
    """

    chat: tuple[str, ...] = ()
    embeddings: tuple[str, ...] = ()
    stream: tuple[str, ...] = ()

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
