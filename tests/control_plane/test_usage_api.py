from fastapi.testclient import TestClient

from app.control_plane.app import app
from app.control_plane.usage.models import UsageEvent

client = TestClient(app)


def _record_event(
    usage_store,
    *,
    request_id: str = "req-123",
    capability: str = "llm.chat",
    status: str = "success",
) -> UsageEvent:
    event = UsageEvent(
        request_id=request_id,
        capability=capability,
        provider="openai",
        model="gpt-4.1-mini",
        tokens_in=10,
        tokens_out=20,
        estimated_cost=0.001,
        latency_ms=50,
        status=status,
    )
    usage_store.record(event)
    return event


def test_usage_requires_api_key(usage_store) -> None:
    response = client.get("/api/v1/platform/usage")

    assert response.status_code == 401


def test_usage_returns_events(usage_store) -> None:
    _record_event(usage_store)

    response = client.get(
        "/api/v1/platform/usage",
        headers={"x-api-key": "super-secret-key"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 1
    assert len(body["events"]) == 1
    assert body["events"][0]["request_id"] == "req-123"
    assert body["events"][0]["capability"] == "llm.chat"
    assert body["events"][0]["provider"] == "openai"
    assert body["events"][0]["model"] == "gpt-4.1-mini"


def test_usage_filters_by_request_id(usage_store) -> None:
    _record_event(usage_store, request_id="req-123")
    _record_event(usage_store, request_id="req-456")

    response = client.get(
        "/api/v1/platform/usage?request_id=req-123",
        headers={"x-api-key": "super-secret-key"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 1
    assert body["events"][0]["request_id"] == "req-123"


def test_usage_filters_by_capability(usage_store) -> None:
    _record_event(usage_store, capability="llm.chat")
    _record_event(usage_store, capability="llm.embeddings")

    response = client.get(
        "/api/v1/platform/usage?capability=llm.embeddings",
        headers={"x-api-key": "super-secret-key"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 1
    assert body["events"][0]["capability"] == "llm.embeddings"


def test_usage_filters_by_status(usage_store) -> None:
    _record_event(usage_store, status="success")
    _record_event(usage_store, status="error")

    response = client.get(
        "/api/v1/platform/usage?status=error",
        headers={"x-api-key": "super-secret-key"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 1
    assert body["events"][0]["status"] == "error"


def test_usage_respects_limit(usage_store) -> None:
    for index in range(5):
        _record_event(usage_store, request_id=f"req-{index}")

    response = client.get(
        "/api/v1/platform/usage?limit=2",
        headers={"x-api-key": "super-secret-key"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 2
    assert len(body["events"]) == 2
    assert body["events"][0]["request_id"] == "req-3"
    assert body["events"][1]["request_id"] == "req-4"
