from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.control_plane.app import app
from app.control_plane.dependencies import (
    get_usage_store,
)

client = TestClient(app)

AUTH_HEADERS = {"x-api-key": "super-secret-key"}


def setup_function() -> None:
    get_usage_store().clear()


def test_chat_route_rejects_missing_required_fields() -> None:
    response = client.post(
        "/api/v1/llm/chat",
        json={},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_embeddings_route_rejects_missing_required_fields() -> None:
    response = client.post(
        "/api/v1/llm/embeddings",
        json={},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_chat_records_usage_event() -> None:
    fake_router = AsyncMock()
    fake_router.route_chat_with_metadata.return_value = type(
        "ChatResult",
        (),
        {
            "response": {
                "reply": "Hello from the model",
                "usage": {
                    "tokens_in": 10,
                    "tokens_out": 20,
                },
            },
            "provider_name": "openai",
            "model_name": "gpt-4.1-mini",
        },
    )()

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    body = response.json()

    assert body["reply"] == "Hello from the model"

    request_id = body["metrics"]["request_id"]

    events = get_usage_store().list(
        request_id=request_id,
    )

    assert len(events) == 1

    event = events[0]

    assert event.request_id == request_id
    assert event.capability == "llm.chat"
    assert event.provider == "openai"
    assert event.model == "gpt-4.1-mini"
    assert event.tokens_in == 10
    assert event.tokens_out == 20
    assert event.status == "success"
    assert event.latency_ms >= 0


def test_chat_records_actual_fallback_provider() -> None:
    fake_router = AsyncMock()
    fake_router.route_chat_with_metadata.return_value = type(
        "ChatResult",
        (),
        {
            "response": {
                "reply": "Fallback response",
                "usage": {
                    "tokens_in": 5,
                    "tokens_out": 8,
                },
            },
            "provider_name": "fallback-provider",
            "model_name": "gpt-4.1-mini",
        },
    )()

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    request_id = response.json()["metrics"]["request_id"]

    events = get_usage_store().list(
        request_id=request_id,
    )

    assert len(events) == 1
    assert events[0].provider == "fallback-provider"


def test_chat_records_error_usage_event() -> None:
    fake_router = AsyncMock()
    fake_router.route_chat_with_metadata.side_effect = ValueError("Invalid chat request")

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 400

    events = get_usage_store().list()

    assert len(events) == 1

    event = events[0]

    assert event.capability == "llm.chat"
    assert event.provider == "openai"
    assert event.model == "gpt-4.1-mini"
    assert event.status == "error"
    assert event.tokens_in == 0
    assert event.tokens_out == 0


def test_embeddings_records_usage_event() -> None:
    fake_router = AsyncMock()
    fake_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.1, 0.2, 0.3],
            "provider_name": "openai",
        },
    )()

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    body = response.json()

    assert body["vector"] == [0.1, 0.2, 0.3]

    request_id = body["metrics"]["request_id"]

    events = get_usage_store().list(
        request_id=request_id,
    )

    assert len(events) == 1

    event = events[0]

    assert event.request_id == request_id
    assert event.capability == "llm.embeddings"
    assert event.provider == "openai"
    assert event.model == "gpt-4.1-mini"
    assert event.tokens_in == 2
    assert event.tokens_out == 0
    assert event.status == "success"
    assert event.latency_ms >= 0


def test_embeddings_records_actual_fallback_provider() -> None:
    fake_router = AsyncMock()
    fake_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.9, 0.8, 0.7],
            "provider_name": "fallback-provider",
        },
    )()

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    request_id = response.json()["metrics"]["request_id"]

    events = get_usage_store().list(
        request_id=request_id,
    )

    assert len(events) == 1
    assert events[0].provider == "fallback-provider"


def test_embeddings_records_error_usage_event() -> None:
    fake_router = AsyncMock()
    fake_router.route_embeddings_with_metadata.side_effect = ValueError(
        "Embedding provider failure",
    )

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 404

    events = get_usage_store().list()

    assert len(events) == 1

    event = events[0]

    assert event.capability == "llm.embeddings"
    assert event.provider == "openai"
    assert event.model == "gpt-4.1-mini"
    assert event.status == "error"
