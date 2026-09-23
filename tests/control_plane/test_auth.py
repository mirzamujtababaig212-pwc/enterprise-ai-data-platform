import hashlib

from fastapi.testclient import TestClient

from app.control_plane.app import app
from app.control_plane.auth import (
    identity_from_api_key,
    principal_from_api_key,
    tenant_id_from_api_key,
)

client = TestClient(app)


def auth_headers():
    return {"x-api-key": "super-secret-key"}


def test_principal_from_api_key_returns_deterministic_fingerprint() -> None:
    api_key = "super-secret-key"

    principal = principal_from_api_key(api_key)

    expected_digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    assert principal == f"api_key:{expected_digest}"
    assert api_key not in principal


def test_authenticated_request_does_not_expose_raw_api_key_as_principal() -> None:
    response = client.get(
        "/api/v1/platform/capabilities",
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert "super-secret-key" not in response.text


def test_health_is_public():
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_platform_health_is_public():
    response = client.get("/api/v1/platform/health")

    assert response.status_code == 200


def test_platform_capabilities_requires_authentication():
    response = client.get("/api/v1/platform/capabilities")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_platform_capabilities_rejects_invalid_key():
    response = client.get(
        "/api/v1/platform/capabilities",
        headers={"x-api-key": "wrong-key"},
    )

    assert response.status_code == 401


def test_platform_capabilities_accepts_valid_key():
    response = client.get(
        "/api/v1/platform/capabilities",
        headers=auth_headers(),
    )

    assert response.status_code == 200


def test_llm_chat_requires_authentication():
    response = client.post(
        "/api/v1/llm/chat",
        json={
            "prompt": "Explain RAG.",
            "provider": "mock",
            "model": "mock-gpt",
            "temperature": 0.2,
            "max_tokens": 100,
            "stream": False,
        },
    )

    assert response.status_code == 401


def test_llm_chat_rejects_invalid_key():
    response = client.post(
        "/api/v1/llm/chat",
        json={
            "prompt": "Explain RAG.",
            "provider": "mock",
            "model": "mock-gpt",
            "temperature": 0.2,
            "max_tokens": 100,
            "stream": False,
        },
        headers={"x-api-key": "wrong-key"},
    )

    assert response.status_code == 401


def test_ml_requires_authentication():
    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json={
            "event_count": 10,
            "avg_speed": 50,
            "max_speed": 80,
            "speed_stddev": 5,
            "avg_rpm": 2000,
            "max_rpm": 3000,
            "avg_fuel_level": 50,
            "min_fuel_level": 40,
            "avg_battery": 12,
            "avg_engine_temperature": 90,
            "max_engine_temperature": 100,
        },
    )

    assert response.status_code == 401


def test_rag_query_requires_authentication():
    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is the enterprise AI platform?",
            "top_k": 5,
        },
    )

    assert response.status_code == 401


def test_rag_index_requires_authentication():
    response = client.post(
        "/api/v1/rag/index",
        json={
            "document_id": "auth-test-document",
            "content": "Authentication test document.",
        },
    )

    assert response.status_code == 401


def test_tenant_id_from_api_key_uses_explicit_mapping(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.control_plane.auth.settings.CONTROL_PLANE_API_KEY_TENANTS",
        "tenant-a=super-secret-key,tenant-b=other-secret-key",
    )

    assert tenant_id_from_api_key("super-secret-key") == "tenant-a"


def test_tenant_id_from_api_key_rejects_unmapped_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.control_plane.auth.settings.CONTROL_PLANE_API_KEY_TENANTS",
        "tenant-a=super-secret-key",
    )

    import pytest

    with pytest.raises(ValueError, match="no configured tenant"):
        tenant_id_from_api_key("other-secret-key")


def test_identity_from_api_key_contains_no_raw_secret(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.control_plane.auth.settings.CONTROL_PLANE_API_KEY_TENANTS",
        "tenant-a=super-secret-key",
    )

    identity = identity_from_api_key("super-secret-key")

    assert identity.tenant_id == "tenant-a"
    assert identity.principal.startswith("api_key:")
    assert identity.user_id is None
    assert "super-secret-key" not in repr(identity)
