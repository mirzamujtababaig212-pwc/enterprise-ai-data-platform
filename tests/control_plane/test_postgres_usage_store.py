from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import Base
from app.control_plane.usage.models import UsageEvent
from app.control_plane.usage.postgres_store import PostgreSQLUsageRepository


@pytest.fixture()
def repository():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    session = session_factory()

    try:
        yield PostgreSQLUsageRepository(session)
    finally:
        session.close()
        engine.dispose()


def make_event(
    *,
    event_id: str = "event-1",
    request_id: str = "request-1",
    capability: str = "llm.chat",
    status: str = "success",
) -> UsageEvent:
    return UsageEvent(
        event_id=event_id,
        request_id=request_id,
        timestamp=datetime.now(UTC),
        capability=capability,
        provider="openai",
        model="gpt-4o",
        tokens_in=100,
        tokens_out=50,
        estimated_cost=0.01,
        latency_ms=125.5,
        status=status,
    )


def test_record_and_get(repository):
    event = make_event()

    repository.record(event)

    result = repository.get(event.event_id)

    assert result is not None
    assert result.event_id == event.event_id
    assert result.request_id == event.request_id
    assert result.capability == event.capability
    assert result.provider == event.provider
    assert result.model == event.model
    assert result.tokens_in == 100
    assert result.tokens_out == 50
    assert result.status == "success"


def test_get_missing_event_returns_none(repository):
    assert repository.get("does-not-exist") is None


def test_list_returns_events(repository):
    repository.record(make_event(event_id="event-1"))
    repository.record(
        make_event(
            event_id="event-2",
            request_id="request-2",
            capability="rag.query",
        )
    )

    events = repository.list()

    assert len(events) == 2
    assert events[0].event_id == "event-1"
    assert events[1].event_id == "event-2"


def test_list_filters_by_request_id(repository):
    repository.record(make_event(event_id="event-1", request_id="request-1"))
    repository.record(make_event(event_id="event-2", request_id="request-2"))

    events = repository.list(request_id="request-1")

    assert len(events) == 1
    assert events[0].event_id == "event-1"


def test_list_filters_by_capability(repository):
    repository.record(make_event(event_id="event-1", capability="llm.chat"))
    repository.record(make_event(event_id="event-2", capability="rag.query"))

    events = repository.list(capability="rag.query")

    assert len(events) == 1
    assert events[0].capability == "rag.query"


def test_list_filters_by_status(repository):
    repository.record(make_event(event_id="event-1", status="success"))
    repository.record(make_event(event_id="event-2", status="error"))

    events = repository.list(status="error")

    assert len(events) == 1
    assert events[0].status == "error"


def test_list_respects_limit(repository):
    for index in range(5):
        repository.record(make_event(event_id=f"event-{index}"))

    events = repository.list(limit=2)

    assert len(events) == 2


def test_clear_removes_events(repository):
    repository.record(make_event())

    repository.clear()

    assert repository.list() == []


def test_implements_usage_repository_contract(repository):
    from app.control_plane.usage.repository import UsageRepository

    contract: UsageRepository = repository

    assert contract is not None
