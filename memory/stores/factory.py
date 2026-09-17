from __future__ import annotations

from common.config.settings import Settings
from memory.contracts import MemoryStore
from memory.stores.in_memory import InMemoryMemoryStore
from memory.stores.postgres import PostgreSQLMemoryStore


class MemoryStoreFactory:
    """
    Constructs the configured memory-store backend.

    In-memory storage remains the default so existing local/test behavior
    is unchanged unless MEMORY_STORE_BACKEND is explicitly set to postgres.
    """

    @staticmethod
    def type_name(backend: str) -> str:
        backend = backend.strip().lower()

        type_names = {
            "in_memory": "InMemoryMemoryStore",
            "postgres": "PostgreSQLMemoryStore",
        }

        try:
            return type_names[backend]
        except KeyError as exc:
            raise ValueError(
                "Unsupported memory-store backend: "
                f"{backend!r}. Expected 'in_memory' or 'postgres'."
            ) from exc

    @staticmethod
    def create(backend: str | None = None) -> MemoryStore:
        backend = Settings.memory_store.BACKEND if backend is None else backend.strip().lower()

        if backend == "in_memory":
            return InMemoryMemoryStore()

        if backend == "postgres":
            return PostgreSQLMemoryStore()

        raise ValueError(
            "Unsupported memory-store backend: " f"{backend!r}. Expected 'in_memory' or 'postgres'."
        )
