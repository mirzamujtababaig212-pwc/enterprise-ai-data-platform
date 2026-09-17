from __future__ import annotations

import pytest

from common.config.settings import Settings
from memory.stores.factory import MemoryStoreFactory
from memory.stores.in_memory import InMemoryMemoryStore
from memory.stores.postgres import PostgreSQLMemoryStore


def test_factory_defaults_to_in_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        Settings.memory_store,
        "BACKEND",
        "in_memory",
    )

    store = MemoryStoreFactory.create()

    assert isinstance(store, InMemoryMemoryStore)


def test_factory_creates_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        Settings.memory_store,
        "BACKEND",
        "postgres",
    )

    store = MemoryStoreFactory.create()

    assert isinstance(store, PostgreSQLMemoryStore)


def test_factory_explicit_backend_without_using_config(monkeypatch) -> None:
    monkeypatch.setattr(
        Settings.memory_store,
        "BACKEND",
        "in_memory",
    )

    store = MemoryStoreFactory.create(
        backend="postgres",
    )

    assert isinstance(store, PostgreSQLMemoryStore)


def test_factory_type_name() -> None:
    assert MemoryStoreFactory.type_name("in_memory") == "InMemoryMemoryStore"
    assert MemoryStoreFactory.type_name("postgres") == "PostgreSQLMemoryStore"


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported memory-store backend",
    ):
        MemoryStoreFactory.create(
            backend="unsupported",
        )
