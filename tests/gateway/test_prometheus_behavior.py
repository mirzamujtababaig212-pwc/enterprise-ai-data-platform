"""
Behavioral tests for the canonical Prometheus metrics used by the
Enterprise AI Gateway.

These tests verify that the canonical collectors are wired into the
HTTP request lifecycle and that /metrics exposes their recorded values.

Important:
- /metrics is scrapeable without authentication.
- Gateway endpoints are currently protected by API-key authentication.
- Therefore an unauthenticated request is expected to produce HTTP 401.
"""

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


HEADERS = {
    "x-api-key": "super-secret-key",
}


# ============================================================================
# PROMETHEUS HELPERS
# ============================================================================


def _metric_value(
    body: str,
    metric_name: str,
    labels: dict[str, str],
) -> float:
    """
    Extract a labeled Prometheus metric value.

    Prometheus does not guarantee that labels will appear in the same
    order in which they were declared in the Python collector. Therefore
    label matching is intentionally order-independent.
    """

    prefix = f"{metric_name}{{"

    for line in body.splitlines():
        if not line.startswith(prefix):
            continue

        metric_labels = line[len(prefix) :].split("}", 1)[0]

        actual_labels = {}

        for item in metric_labels.split(","):
            key, value = item.split("=", 1)
            actual_labels[key] = value.strip('"')

        if all(actual_labels.get(key) == value for key, value in labels.items()):
            return float(line.split()[-1])

    matching_lines = "\n".join(line for line in body.splitlines() if line.startswith(metric_name))

    raise AssertionError(
        f"Metric {metric_name!r} with labels "
        f"{labels!r} was not found in Prometheus output.\n\n"
        f"Available matching lines:\n"
        f"{matching_lines}"
    )


def _metric_exists(
    body: str,
    metric_name: str,
    labels: dict[str, str],
) -> bool:
    """
    Return True when a Prometheus metric with the requested labels exists.

    Label order is intentionally ignored.
    """

    prefix = f"{metric_name}{{"

    for line in body.splitlines():
        if not line.startswith(prefix):
            continue

        metric_labels = line[len(prefix) :].split("}", 1)[0]

        actual_labels = {}

        for item in metric_labels.split(","):
            key, value = item.split("=", 1)

            actual_labels[key] = value.strip('"')

        if all(actual_labels.get(key) == value for key, value in labels.items()):
            return True

    return False


# ============================================================================
# HTTP REQUEST COUNTER
# ============================================================================


def test_http_request_counter_increases():
    """
    Prove that an authenticated HTTP request increments the canonical
    HTTP request counter.
    """

    before_response = client.get("/metrics")

    assert before_response.status_code == 200

    body_before = before_response.text

    try:
        before_value = _metric_value(
            body_before,
            "llm_gateway_http_requests_total",
            {
                "method": "GET",
                "endpoint": "/v1/health",
                "status_code": "200",
            },
        )

    except AssertionError:
        before_value = 0.0

    # ------------------------------------------------------------------
    # Perform the observed request.
    # ------------------------------------------------------------------

    observed_response = client.get(
        "/v1/health",
        headers=HEADERS,
    )

    assert observed_response.status_code == 200

    # ------------------------------------------------------------------
    # Scrape metrics after the request.
    # ------------------------------------------------------------------

    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200

    after_value = _metric_value(
        metrics_response.text,
        "llm_gateway_http_requests_total",
        {
            "method": "GET",
            "endpoint": "/v1/health",
            "status_code": "200",
        },
    )

    assert after_value > before_value


# ============================================================================
# HTTP REQUEST DURATION
# ============================================================================


def test_http_request_duration_is_recorded():
    """
    Prove that an authenticated HTTP request contributes to the canonical
    request-duration histogram.
    """

    response = client.get(
        "/v1/health",
        headers=HEADERS,
    )

    assert response.status_code == 200

    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200

    body = metrics_response.text

    assert _metric_exists(
        body,
        "llm_gateway_http_request_duration_seconds_count",
        {
            "method": "GET",
            "endpoint": "/v1/health",
            "status_code": "200",
        },
    )


# ============================================================================
# UNAUTHENTICATED REQUEST / 401
# ============================================================================


def test_unauthenticated_request_is_recorded_as_error():
    """
    Prove that the authentication middleware's 401 response is observed
    by the canonical HTTP error counter.

    RequestLoggingMiddleware is intentionally the outermost middleware,
    allowing it to observe authentication-generated 401 responses.
    """

    before_response = client.get("/metrics")

    assert before_response.status_code == 200

    body_before = before_response.text

    try:
        before_value = _metric_value(
            body_before,
            "llm_gateway_errors_total",
            {
                "status_code": "401",
            },
        )

    except AssertionError:
        before_value = 0.0

    # ------------------------------------------------------------------
    # Deliberately omit the API key.
    # ------------------------------------------------------------------

    unauthorized_response = client.get(
        "/this-route-does-not-exist",
    )

    assert unauthorized_response.status_code == 401

    # ------------------------------------------------------------------
    # Scrape metrics after the unauthorized request.
    # ------------------------------------------------------------------

    after_response = client.get("/metrics")

    assert after_response.status_code == 200

    after_value = _metric_value(
        after_response.text,
        "llm_gateway_errors_total",
        {
            "status_code": "401",
        },
    )

    assert after_value > before_value


# ============================================================================
# AUTHENTICATED SUCCESS MUST NOT BE AN ERROR
# ============================================================================


def test_authenticated_health_request_is_not_recorded_as_error():
    """
    Prove that a successful authenticated request does not increment
    the HTTP error counter for status 200.
    """

    before_response = client.get("/metrics")

    assert before_response.status_code == 200

    body_before = before_response.text

    try:
        before_value = _metric_value(
            body_before,
            "llm_gateway_errors_total",
            {
                "status_code": "200",
            },
        )

    except AssertionError:
        before_value = 0.0

    response = client.get(
        "/v1/health",
        headers=HEADERS,
    )

    assert response.status_code == 200

    after_response = client.get("/metrics")

    assert after_response.status_code == 200

    body_after = after_response.text

    try:
        after_value = _metric_value(
            body_after,
            "llm_gateway_errors_total",
            {
                "status_code": "200",
            },
        )

    except AssertionError:
        after_value = 0.0

    assert after_value == before_value
