from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


VALID_API_KEY = "super-secret-key"


def test_missing_api_key_is_rejected():
    response = client.get("/v1/health")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_invalid_api_key_is_rejected():
    response = client.get(
        "/v1/health",
        headers={
            "x-api-key": "invalid-key",
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_valid_api_key_is_accepted():
    response = client.get(
        "/v1/health",
        headers={
            "x-api-key": VALID_API_KEY,
        },
    )

    assert response.status_code == 200


def test_metrics_endpoint_does_not_require_api_key():
    response = client.get("/metrics")

    assert response.status_code == 200


def test_openapi_endpoint_does_not_require_api_key():
    response = client.get("/openapi.json")

    assert response.status_code == 200


def test_docs_endpoint_does_not_require_api_key():
    response = client.get("/docs")

    assert response.status_code == 200


def test_redoc_endpoint_does_not_require_api_key():
    response = client.get("/redoc")

    assert response.status_code == 200


def test_favicon_endpoint_does_not_require_api_key():
    response = client.get("/favicon.ico")

    assert response.status_code != 401
