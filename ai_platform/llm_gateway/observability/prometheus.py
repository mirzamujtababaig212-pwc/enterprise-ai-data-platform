"""
Backward-compatible Prometheus metric imports.

The canonical metric definitions live in:

    ai_platform.llm_gateway.metrics.prometheus

This module intentionally defines NO Prometheus collectors.
"""

from ai_platform.llm_gateway.metrics.prometheus import (
    ERRORS_TOTAL,
    ESTIMATED_COST_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    INPUT_TOKENS_TOTAL,
    MODEL_REQUESTS_TOTAL,
    OUTPUT_TOKENS_TOTAL,
    PROVIDER_REQUESTS_TOTAL,
)

__all__ = [
    "ERRORS_TOTAL",
    "ESTIMATED_COST_TOTAL",
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "INPUT_TOKENS_TOTAL",
    "MODEL_REQUESTS_TOTAL",
    "OUTPUT_TOKENS_TOTAL",
    "PROVIDER_REQUESTS_TOTAL",
]
