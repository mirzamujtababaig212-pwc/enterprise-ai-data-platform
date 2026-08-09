"""
Canonical Prometheus metrics for the Enterprise AI Gateway.

All gateway Prometheus collectors are defined in this module.

IMPORTANT:
    Do not define the same collectors anywhere else.
    ai_platform.llm_gateway.observability.prometheus only re-exports
    these objects for backward compatibility.
"""

from prometheus_client import Counter, Histogram

# ============================================================================
# HTTP METRICS
# ============================================================================

HTTP_REQUESTS_TOTAL = Counter(
    "llm_gateway_http_requests_total",
    "Total HTTP requests processed.",
    [
        "method",
        "endpoint",
        "status_code",
    ],
)


HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "llm_gateway_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    [
        "method",
        "endpoint",
        "status_code",
    ],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
    ),
)


# ============================================================================
# PROVIDER METRICS
# ============================================================================

PROVIDER_REQUESTS_TOTAL = Counter(
    "llm_gateway_provider_requests_total",
    "Total requests by provider.",
    [
        "provider",
    ],
)


MODEL_REQUESTS_TOTAL = Counter(
    "llm_gateway_model_requests_total",
    "Total requests by model.",
    [
        "model",
    ],
)


# ============================================================================
# TOKEN METRICS
# ============================================================================

INPUT_TOKENS_TOTAL = Counter(
    "llm_gateway_input_tokens_total",
    "Total input tokens.",
)


OUTPUT_TOKENS_TOTAL = Counter(
    "llm_gateway_output_tokens_total",
    "Total output tokens.",
)


# ============================================================================
# COST METRICS
# ============================================================================

ESTIMATED_COST_TOTAL = Counter(
    "llm_gateway_estimated_cost_total",
    "Estimated LLM usage cost.",
)


# ============================================================================
# ERROR METRICS
# ============================================================================

ERRORS_TOTAL = Counter(
    "llm_gateway_errors_total",
    "Total HTTP errors produced by the LLM Gateway.",
    [
        "status_code",
    ],
)
