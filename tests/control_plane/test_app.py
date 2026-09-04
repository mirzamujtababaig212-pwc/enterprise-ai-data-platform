from fastapi.testclient import TestClient

from app.control_plane.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "enterprise-ai-control-plane",
        "version": "1.0.0",
    }


def test_platform_health() -> None:
    response = client.get("/api/v1/platform/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_platform_capabilities() -> None:
    response = client.get("/api/v1/platform/capabilities")

    assert response.status_code == 200

    payload = response.json()

    assert payload["service"] == "enterprise-ai-platform"

    capability_names = {capability["name"] for capability in payload["capabilities"]}

    assert "llm.chat" in capability_names
    assert "llm.embeddings" in capability_names
    assert "ml.vehicle-risk" in capability_names
