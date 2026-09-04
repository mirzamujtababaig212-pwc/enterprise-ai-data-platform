from fastapi.testclient import TestClient

from app.control_plane.app import app

client = TestClient(app)


def test_chat_route_rejects_missing_required_fields() -> None:
    response = client.post(
        "/api/v1/llm/chat",
        json={},
    )

    assert response.status_code == 422


def test_embeddings_route_rejects_missing_required_fields() -> None:
    response = client.post(
        "/api/v1/llm/embeddings",
        json={},
    )

    assert response.status_code == 422
