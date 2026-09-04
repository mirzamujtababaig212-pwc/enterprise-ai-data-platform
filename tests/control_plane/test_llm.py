from fastapi.testclient import TestClient

from app.control_plane.app import app

client = TestClient(app)

AUTH_HEADERS = {"x-api-key": "super-secret-key"}


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
