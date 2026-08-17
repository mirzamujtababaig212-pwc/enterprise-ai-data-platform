from __future__ import annotations

from abc import ABC, abstractmethod

BytesLike = bytes | bytearray


class BaseStorage(ABC):
    """
    Common interface implemented by all storage backends.
    """

    @abstractmethod
    def write(self, key: str, data: BytesLike) -> None:
        """
        Write bytes to the storage backend.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, key: str) -> bytes:
        """
        Read bytes from the storage backend.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Return whether an object exists.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete an object.
        """
        raise NotImplementedError

    @abstractmethod
    def uri(self, key: str) -> str:
        """
        Return a URI representing the object.
        """
        raise NotImplementedError
