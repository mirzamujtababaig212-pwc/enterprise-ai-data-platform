"""
Prometheus metrics for the Enterprise AI Gateway.

This module defines all Prometheus counters and histograms used by
the gateway and exposes helper functions for recording metrics.

All metrics are registered once when this module is imported.
"""

from prometheus_client import Counter
from prometheus_client import Histogram

#
# HTTP Metrics
#

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
    "HTTP request latency.",
    [
        "method",
        "endpoint",
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

#
# Provider Metrics
#

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

#
# Token Metrics
#

INPUT_TOKENS_TOTAL = Counter(
    "llm_gateway_input_tokens_total",
    "Total input tokens.",
)

OUTPUT_TOKENS_TOTAL = Counter(
    "llm_gateway_output_tokens_total",
    "Total output tokens.",
)

#
# Estimated Cost
#

ESTIMATED_COST_TOTAL = Counter(
    "llm_gateway_estimated_cost_total",
    "Estimated LLM usage cost.",
)

#
# Errors
#

ERRORS_TOTAL = Counter(
    "llm_gateway_errors_total",
    "Gateway errors.",
    [
        "status_code",
    ],
)
