from typing import Any


class ModelRegistry:
    """
    Runtime registry of provider capabilities.

    Provider implementations are the source of truth for the models they
    support. The registry discovers those capabilities when providers are
    registered.

    Capability names currently supported:
        - chat
        - embeddings
        - stream
    """

    def __init__(self):
        self._models: dict[str, dict[str, list[str]]] = {}

    def register_provider(
        self,
        provider_name: str,
        provider: Any,
    ) -> None:
        """
        Register the capabilities exposed by a provider.
        """

        chat_models = []

        embedding_models = []
        if hasattr(provider, "supported_chat_models"):
            chat_models = list(provider.supported_chat_models())

        if hasattr(provider, "supported_embedding_models"):
            embedding_models = list(provider.supported_embedding_models())

        self._models[provider_name] = {
            "chat": chat_models,
            "embeddings": embedding_models,
        }

        stream_models = self._get_supported_models(
            provider,
            "supported_stream_models",
        )

        # Backward-compatible behavior:
        #
        # All current providers implement stream() but do not yet expose
        # supported_stream_models(). In that case, stream capability follows
        # chat capability.
        if not stream_models and hasattr(provider, "stream"):
            stream_models = list(chat_models)

        self._models[provider_name] = {
            "chat": sorted(set(chat_models)),
            "embeddings": sorted(set(embedding_models)),
            "stream": sorted(set(stream_models)),
        }

    @staticmethod
    def _get_supported_models(
        provider: Any,
        method_name: str,
    ) -> list[str]:
        """
        Safely obtain a provider's supported model list.
        """

        method = getattr(provider, method_name, None)

        if method is None:
            return []

        models = method()

        if models is None:
            return []

        return list(models)

    def unregister_provider(
        self,
        provider_name: str,
    ) -> None:
        """
        Remove a provider from the registry.
        """

        self._models.pop(
            provider_name,
            None,
        )

    def provider_exists(
        self,
        provider_name: str,
    ) -> bool:
        """
        Return whether a provider is registered.
        """

        return provider_name in self._models

    def model_supported(
        self,
        provider_name: str,
        capability: str,
        model: str,
    ) -> bool:
        """
        Check whether a provider supports a model for a capability.
        """

        provider_models = self._models.get(
            provider_name,
            {},
        )

        return model in provider_models.get(
            capability,
            [],
        )

    def get_models(
        self,
        provider_name: str,
    ) -> dict[str, list[str]]:
        """
        Return all capabilities for a provider.
        """

        return self._models.get(
            provider_name,
            {},
        )

    def get_providers_for_model(
        self,
        capability: str,
        model: str,
    ) -> list[str]:
        """
        Return providers supporting a model for a capability.
        """

        providers = []

        for provider_name, capabilities in self._models.items():
            models = capabilities.get(
                capability,
                [],
            )

            if model in models:
                providers.append(provider_name)

        return providers

    def list_providers(self) -> list[str]:
        """
        Return registered providers.
        """

        return list(self._models.keys())

    def list_models(
        self,
        capability: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Return registered models.

        If capability is supplied, return only that capability.

        Otherwise, return the union of all models exposed by each provider.
        """

        result: dict[str, list[str]] = {}

        for provider_name, capabilities in self._models.items():

            if capability is None:
                models = []

                for provider_models in capabilities.values():
                    models.extend(provider_models)

                result[provider_name] = sorted(set(models))

            else:
                result[provider_name] = sorted(
                    set(
                        capabilities.get(
                            capability,
                            [],
                        )
                    )
                )

        return result

    def clear(self) -> None:
        """
        Remove all registered providers and capabilities.
        """

        self._models.clear()


model_registry = ModelRegistry()
