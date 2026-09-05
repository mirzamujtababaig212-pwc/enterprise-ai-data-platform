import pytest

from app.control_plane.app import app
from app.control_plane.dependencies import get_usage_store
from app.control_plane.usage.store import InMemoryUsageStore


@pytest.fixture
def usage_store():
    store = InMemoryUsageStore()

    previous_override = app.dependency_overrides.get(get_usage_store)

    app.dependency_overrides[get_usage_store] = lambda: store

    try:
        yield store
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_usage_store, None)
        else:
            app.dependency_overrides[get_usage_store] = previous_override
