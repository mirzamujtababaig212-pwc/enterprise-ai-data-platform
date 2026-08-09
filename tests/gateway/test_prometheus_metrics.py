from prometheus_client import REGISTRY

from ai_platform.llm_gateway.metrics import prometheus

EXPECTED_METRICS = {
    "llm_gateway_http_requests",
    "llm_gateway_http_request_duration_seconds",
    "llm_gateway_provider_requests",
    "llm_gateway_model_requests",
    "llm_gateway_input_tokens",
    "llm_gateway_output_tokens",
    "llm_gateway_estimated_cost",
    "llm_gateway_errors",
}


def test_gateway_prometheus_metrics_are_registered():
    found = {metric.name for metric in REGISTRY.collect() if metric.name.startswith("llm_gateway")}

    missing = EXPECTED_METRICS - found

    assert not missing, "Expected gateway Prometheus metrics are missing: " f"{sorted(missing)}"


def test_canonical_prometheus_module_exports_metrics():
    assert prometheus.HTTP_REQUESTS_TOTAL is not None
    assert prometheus.HTTP_REQUEST_DURATION_SECONDS is not None
    assert prometheus.PROVIDER_REQUESTS_TOTAL is not None
    assert prometheus.MODEL_REQUESTS_TOTAL is not None
    assert prometheus.INPUT_TOKENS_TOTAL is not None
    assert prometheus.OUTPUT_TOKENS_TOTAL is not None
    assert prometheus.ESTIMATED_COST_TOTAL is not None
    assert prometheus.ERRORS_TOTAL is not None
