from abc import ABC, abstractmethod


class BaseProvider(ABC):

    @abstractmethod
    def chat(self, request: dict) -> dict: ...

    @abstractmethod
    def stream(self, request: dict) -> dict: ...

    @abstractmethod
    def embeddings(self, request: dict) -> list[float]: ...

    @abstractmethod
    def health_check(self) -> dict: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...
