from fastapi.testclient import TestClient

from app.control_plane.app import app

client = TestClient(app)

AUTH_HEADERS = {"x-api-key": "super-secret-key"}


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
    response = client.get("/api/v1/platform/capabilities", headers=AUTH_HEADERS)

    assert response.status_code == 200

    payload = response.json()

    assert payload["service"] == "enterprise-ai-platform"

    capability_names = {capability["name"] for capability in payload["capabilities"]}

    assert "llm.chat" in capability_names
    assert "llm.embeddings" in capability_names
    assert "ml.vehicle-risk" in capability_names


def test_lifespan_closes_rag_vector_store(monkeypatch) -> None:
    calls: list[bool] = []

    async def fake_close() -> None:
        calls.append(True)

    monkeypatch.setattr(
        "app.control_plane.app.close_rag_vector_store",
        fake_close,
    )

    with TestClient(app):
        pass

    assert calls == [True]
