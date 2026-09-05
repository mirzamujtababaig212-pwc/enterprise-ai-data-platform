from __future__ import annotations

from unittest.mock import Mock

from app.control_plane.dependencies import get_usage_store
from app.control_plane.usage.postgres_store import PostgreSQLUsageRepository


def test_get_usage_store_creates_repository_from_injected_session() -> None:
    session_one = Mock()
    session_two = Mock()

    store_one = get_usage_store(db=session_one)
    store_two = get_usage_store(db=session_two)

    assert isinstance(store_one, PostgreSQLUsageRepository)
    assert isinstance(store_two, PostgreSQLUsageRepository)

    assert store_one is not store_two
    assert store_one._session is session_one
    assert store_two._session is session_two
