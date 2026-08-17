from collections.abc import Iterable
from typing import Protocol


class ProviderProtocol(Protocol):
    """
    Contract implemented by every LLM provider.

    Providers may implement additional methods, but the methods
    below define the minimum platform contract.
    """

    def supported_chat_models(self) -> Iterable[str]: ...

    def supported_embedding_models(self) -> Iterable[str]: ...

    def supported_stream_models(self) -> Iterable[str]: ...

    def chat(self, *args, **kwargs): ...

    def embeddings(self, *args, **kwargs): ...

    def stream(self, *args, **kwargs): ...

    def health(self) -> object: ...
