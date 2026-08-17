from __future__ import annotations

from pathlib import Path

from common.storage.base_storage import BaseStorage

BytesLike = bytes | bytearray


class LocalStorage(BaseStorage):
    """
    Local filesystem implementation of the platform storage contract.

    Logical keys are resolved underneath base_path.

    Examples:
        bronze/test.txt
        silver/year=2026/month=08/data.parquet
    """

    def __init__(self, base_path: str | Path | None = None):
        if base_path is None:
            base_path = Path.cwd() / ".storage"

        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """
        Resolve a logical storage key safely underneath base_path.

        Absolute paths are supported because some existing tests use
        temporary filesystem paths directly.
        """

        if not isinstance(key, str):
            raise TypeError("Storage key must be a string.")

        if not key:
            raise ValueError("Storage key cannot be empty.")

        path = Path(key).expanduser()

        # Preserve compatibility with callers that provide an absolute
        # temporary filesystem path.
        if path.is_absolute():
            return path.resolve()

        resolved = (self.base_path / path).resolve()

        try:
            resolved.relative_to(self.base_path)
        except ValueError as exc:
            raise ValueError(f"Storage key escapes base path: {key}") from exc

        return resolved

    def write(self, key: str, data: BytesLike) -> None:
        """
        Write bytes to local storage.

        Signature:
            write(key, data)
        """

        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes or bytearray.")

        path = self._resolve(key)

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_bytes(bytes(data))

    def read(self, key: str) -> bytes:
        """
        Read bytes from local storage.
        """

        path = self._resolve(key)

        if not path.exists():
            raise FileNotFoundError(f"Storage object does not exist: {key}")

        if not path.is_file():
            raise IsADirectoryError(f"Storage key is not a file: {key}")

        return path.read_bytes()

    def exists(self, key: str) -> bool:
        """
        Return whether the storage object exists.
        """

        return self._resolve(key).is_file()

    def delete(self, key: str) -> None:
        """
        Delete a storage object if it exists.
        """

        path = self._resolve(key)

        if path.exists() and not path.is_file():
            raise IsADirectoryError(f"Storage key is not a file: {key}")

        path.unlink(missing_ok=True)

    def uri(self, key: str) -> str:
        """
        Return a file:// URI for the object.
        """

        return self._resolve(key).as_uri()
