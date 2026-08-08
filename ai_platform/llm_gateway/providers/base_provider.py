from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class BaseProvider(ABC):

    @abstractmethod
    async def chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]: ...

    @abstractmethod
    def stream(
        self,
        request: dict[str, Any],
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]: ...

    @abstractmethod
    async def health_check(
        self,
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def list_models(
        self,
    ) -> list[str]: ...

    @abstractmethod
    def supported_chat_models(
        self,
    ) -> list[str]: ...

    @abstractmethod
    def supported_embedding_models(
        self,
    ) -> list[str]: ...
