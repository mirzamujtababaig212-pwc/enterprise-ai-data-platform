"""
Backward-compatible Prometheus metric imports.

The canonical metric definitions live in:

    ai_platform.llm_gateway.metrics.prometheus

This module intentionally does NOT create any Prometheus collectors.

It only re-exports the canonical collectors so existing imports continue
to work without registering duplicate time series.
"""

from ai_platform.llm_gateway.metrics.prometheus import (
    ERRORS_TOTAL,
    ESTIMATED_COST_TOTAL,
    FALLBACK_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    INPUT_TOKENS_TOTAL,
    MODEL_REQUESTS_TOTAL,
    OUTPUT_TOKENS_TOTAL,
    PROVIDER_ERRORS_TOTAL,
    PROVIDER_LATENCY_SECONDS,
    PROVIDER_REQUESTS_TOTAL,
)

__all__ = [
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "PROVIDER_REQUESTS_TOTAL",
    "PROVIDER_LATENCY_SECONDS",
    "PROVIDER_ERRORS_TOTAL",
    "FALLBACK_REQUESTS_TOTAL",
    "MODEL_REQUESTS_TOTAL",
    "INPUT_TOKENS_TOTAL",
    "OUTPUT_TOKENS_TOTAL",
    "ESTIMATED_COST_TOTAL",
    "ERRORS_TOTAL",
]
