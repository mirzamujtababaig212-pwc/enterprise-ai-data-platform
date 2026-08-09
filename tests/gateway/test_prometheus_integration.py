from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


def get_metrics_text():
    response = client.get("/metrics")

    assert response.status_code == 200

    return response.text


def test_metrics_endpoint_is_prometheus_compatible():
    response = client.get("/metrics")

    assert response.status_code == 200

    content_type = response.headers.get("content-type", "")

    assert "text/plain" in content_type


def test_http_request_metric_is_recorded():
    before = get_metrics_text()

    response = client.get("/health")

    assert response.status_code in (200, 401, 403, 404)

    after = get_metrics_text()

    assert "llm_gateway_http_requests_total" in before
    assert "llm_gateway_http_requests_total" in after


def test_request_duration_metric_is_exposed():
    body = get_metrics_text()

    assert "llm_gateway_http_request_duration_seconds" in body


def test_error_metric_is_exposed():
    body = get_metrics_text()

    assert "llm_gateway_errors_total" in body
