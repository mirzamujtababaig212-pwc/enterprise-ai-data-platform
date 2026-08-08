from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app


client = TestClient(app)

HEADERS = {
    "x-api-key": "super-secret-key",
}


###########################################################################
# Chat endpoint success
###########################################################################


@patch("ai_platform.llm_gateway.api.main.router.route_chat", new_callable=AsyncMock)
def test_chat_success(mock_chat):

    mock_chat.return_value = {
        "reply": "hello",
        "usage": {
            "tokens_in": 5,
            "tokens_out": 3,
        },
    }

    response = client.post(
        "/v1/chat",
        headers=HEADERS,
        json={
            "provider": "openai",
            "model": "gpt-4",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["reply"] == "hello"
    assert body["metrics"]["tokens_in"] == 5
    assert body["metrics"]["tokens_out"] == 3


###########################################################################
# Chat ValueError
###########################################################################


@patch("ai_platform.llm_gateway.api.main.router.route_chat", new_callable=AsyncMock)
def test_chat_value_error(mock_chat):

    mock_chat.side_effect = ValueError("bad provider")

    response = client.post(
        "/v1/chat",
        headers=HEADERS,
        json={
            "provider": "bad",
            "model": "bad",
            "prompt": "hello",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "bad provider"


###########################################################################
# Embeddings success
###########################################################################


@patch(
    "ai_platform.llm_gateway.api.main.router.route_embeddings",
    new_callable=AsyncMock,
)
def test_embeddings_success(mock_embeddings):

    mock_embeddings.return_value = [0.1, 0.2, 0.3]

    response = client.post(
        "/v1/embeddings",
        headers=HEADERS,
        json={
            "provider": "openai",
            "model": "text-embedding-3-small",
            "text": "hello world",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["vector"] == [0.1, 0.2, 0.3]
    assert body["metrics"]["tokens_out"] == 3


###########################################################################
# Embeddings ValueError
###########################################################################


@patch(
    "ai_platform.llm_gateway.api.main.router.route_embeddings",
    new_callable=AsyncMock,
)
def test_embeddings_not_found(mock_embeddings):

    mock_embeddings.side_effect = ValueError("model not found")

    response = client.post(
        "/v1/embeddings",
        headers=HEADERS,
        json={
            "provider": "openai",
            "model": "bad",
            "text": "hello",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "model not found"


###########################################################################
# Models endpoint
###########################################################################


@patch("ai_platform.llm_gateway.api.main.router.route_models", new_callable=AsyncMock)
def test_models(mock_models):

    mock_models.return_value = {
        "openai": [
            "gpt-4",
        ]
    }

    response = client.get(
        "/v1/models",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "openai": [
            "gpt-4",
        ]
    }


###########################################################################
# Health endpoint
###########################################################################


@patch("ai_platform.llm_gateway.api.main.router.route_health", new_callable=AsyncMock)
def test_health(mock_health):
    mock_health.return_value = {"openai": {"status": "healthy"}}  # <-- must be a dict

    response = client.get(
        "/v1/health",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["providers"]["openai"]["status"] == "healthy"


###########################################################################
# Test error endpoint
###########################################################################


def test_test_error():

    response = client.get(
        "/v1/test-error",
        headers=HEADERS,
    )

    assert response.status_code == 500


###########################################################################
# Test body endpoint
###########################################################################


def test_test_body():

    payload = {
        "password": "secret",
        "username": "annie",
    }

    response = client.post(
        "/v1/test-body",
        headers=HEADERS,
        json=payload,
    )

    assert response.status_code == 200

    assert response.json() == {
        "received": payload,
    }


###########################################################################
# Metrics endpoint
###########################################################################


def test_metrics():

    response = client.get(
        "/metrics",
    )

    assert response.status_code == 200

    assert "text/plain" in response.headers["content-type"]
