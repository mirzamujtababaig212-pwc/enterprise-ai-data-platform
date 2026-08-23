from collections.abc import Iterable
from typing import Any

from ai_platform.llm_gateway.models.capabilities import (
    ProviderCapabilities,
    RegistrySnapshot,
)


class ModelRegistry:
    """
    Runtime registry of provider implementations and capabilities.

    The registry maintains:
        1. The actual provider instances.
        2. Capability metadata discovered from those providers.

    Provider implementations remain the source of truth for supported
    models and capabilities.
    """

    def __init__(self):
        self._providers: dict[str, Any] = {}
        self._snapshot = RegistrySnapshot.empty()

    def register_provider(
        self,
        provider_name: str,
        provider: Any,
    ) -> None:
        """
        Register a provider implementation and discover its capabilities.
        """

        self._providers[provider_name] = provider

        capabilities = self._discover_capabilities(
            provider,
        )

        providers = dict(
            self._snapshot.providers,
        )

        providers[provider_name] = capabilities

        self._snapshot = RegistrySnapshot(
            providers=providers,
        )

    def unregister_provider(
        self,
        provider_name: str,
    ) -> None:
        """
        Remove a provider and atomically rebuild the snapshot.
        """

        self._providers.pop(
            provider_name,
            None,
        )

        providers = dict(
            self._snapshot.providers,
        )

        providers.pop(
            provider_name,
            None,
        )

        self._snapshot = RegistrySnapshot(
            providers=providers,
        )

    def get_provider(
        self,
        provider_name: str,
    ) -> Any:
        return self._providers[provider_name]

    def provider_exists(
        self,
        provider_name: str,
    ) -> bool:
        return provider_name in self._providers

    def model_supported(
        self,
        provider_name: str,
        capability: str,
        model: str,
    ) -> bool:
        return self._snapshot.model_supported(
            provider_name,
            capability,
            model,
        )

    def get_models(
        self,
        provider_name: str,
    ) -> dict[str, list[str]]:
        capabilities = self._snapshot.providers.get(
            provider_name,
        )

        if capabilities is None:
            return {}

        return {
            "chat": list(capabilities.chat),
            "embeddings": list(capabilities.embeddings),
            "stream": list(capabilities.stream),
        }

    def get_providers_for_model(
        self,
        capability: str,
        model: str,
    ) -> list[str]:
        return [
            provider
            for provider, capabilities in self._snapshot.providers.items()
            if capabilities.supports(
                capability,
                model,
            )
        ]

    def list_providers(self) -> list[str]:
        return list(
            self._providers.keys(),
        )

    def list_models(
        self,
        capability: str | None = None,
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}

        for (
            provider,
            capabilities,
        ) in self._snapshot.providers.items():
            if capability is None:
                models = sorted(
                    set(
                        capabilities.chat + capabilities.embeddings + capabilities.stream,
                    )
                )
            else:
                models = sorted(
                    capabilities.models_for(
                        capability,
                    )
                )

            result[provider] = models

        return result

    def snapshot(self) -> RegistrySnapshot:
        """
        Return the immutable current capability snapshot.
        """

        return self._snapshot

    def clear(self) -> None:
        self._providers.clear()
        self._snapshot = RegistrySnapshot.empty()

    @classmethod
    def _discover_capabilities(
        cls,
        provider: Any,
    ) -> ProviderCapabilities:

        chat = cls._get_supported_models(
            provider,
            "supported_chat_models",
        )

        embeddings = cls._get_supported_models(
            provider,
            "supported_embedding_models",
        )

        stream = cls._get_supported_models(
            provider,
            "supported_stream_models",
        )

        # Backward compatibility:
        #
        # If a provider exposes stream() but does not explicitly
        # expose supported_stream_models(), inherit chat models.
        if not stream and hasattr(provider, "stream"):
            stream = list(chat)

        return ProviderCapabilities(
            chat=tuple(sorted(set(chat))),
            embeddings=tuple(sorted(set(embeddings))),
            stream=tuple(sorted(set(stream))),
        )

    @staticmethod
    def _get_supported_models(
        provider: Any,
        method_name: str,
    ) -> list[str]:

        method = getattr(
            provider,
            method_name,
            None,
        )

        if method is None:
            return []

        models = method()

        if models is None:
            return []

        return list(
            ModelRegistry._cast_models(models),
        )

    @staticmethod
    def _cast_models(
        models: Iterable[str],
    ) -> Iterable[str]:
        """
        Normalize provider model identifiers.
        """

        for model in models:
            if model is None:
                continue

            value = str(model).strip()

            if value:
                yield value


model_registry = ModelRegistry()
