"""Prometheus metrics for ML inference observability."""

from prometheus_client import Counter, Histogram

ML_INFERENCE_REQUESTS_TOTAL = Counter(
    "ml_inference_requests_total",
    "Total ML inference requests processed.",
    ["model_name", "model_alias", "status"],
)

ML_INFERENCE_DURATION_SECONDS = Histogram(
    "ml_inference_duration_seconds",
    "ML inference duration in seconds.",
    ["model_name", "model_alias"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)

ML_INFERENCE_PREDICTIONS_TOTAL = Counter(
    "ml_inference_predictions_total",
    "Total ML inference predictions by risk category.",
    ["model_name", "model_alias", "risk"],
)

ML_INFERENCE_ERRORS_TOTAL = Counter(
    "ml_inference_errors_total",
    "Total ML inference errors by type.",
    ["model_name", "model_alias", "error_type"],
)
