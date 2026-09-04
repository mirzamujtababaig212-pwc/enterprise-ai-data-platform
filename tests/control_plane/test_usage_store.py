from app.control_plane.usage.models import UsageEvent
from app.control_plane.usage.store import InMemoryUsageStore


def make_event(
    *,
    request_id: str = "req-123",
    capability: str = "llm.chat",
    status: str = "success",
) -> UsageEvent:
    return UsageEvent(
        request_id=request_id,
        capability=capability,
        provider="openai",
        model="gpt-4.1-mini",
        tokens_in=100,
        tokens_out=50,
        estimated_cost=0.00012,
        latency_ms=125,
        status=status,
    )


def test_record_and_list():
    store = InMemoryUsageStore()

    event = make_event()

    recorded = store.record(event)

    assert recorded == event
    assert store.list() == [event]


def test_get_by_event_id():
    store = InMemoryUsageStore()

    event = store.record(make_event())

    assert store.get(event.event_id) == event


def test_get_missing_event_returns_none():
    store = InMemoryUsageStore()

    assert store.get("missing-event") is None


def test_filter_by_request_id():
    store = InMemoryUsageStore()

    first = store.record(
        make_event(request_id="req-1"),
    )
    store.record(
        make_event(request_id="req-2"),
    )

    assert store.list(request_id="req-1") == [first]


def test_filter_by_capability():
    store = InMemoryUsageStore()

    llm_event = store.record(
        make_event(capability="llm.chat"),
    )
    store.record(
        make_event(capability="rag.query"),
    )

    assert store.list(capability="llm.chat") == [llm_event]


def test_filter_by_status():
    store = InMemoryUsageStore()

    success = store.record(
        make_event(status="success"),
    )
    store.record(
        make_event(status="error"),
    )

    assert store.list(status="success") == [success]


def test_limit_returns_latest_events():
    store = InMemoryUsageStore()

    first = store.record(make_event(request_id="req-1"))
    second = store.record(make_event(request_id="req-2"))
    third = store.record(make_event(request_id="req-3"))

    assert store.list(limit=2) == [second, third]
    assert first not in store.list(limit=2)


def test_clear_removes_events():
    store = InMemoryUsageStore()

    store.record(make_event())

    store.clear()

    assert store.list() == []
