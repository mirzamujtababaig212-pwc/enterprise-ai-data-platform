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
    "Total Vehicle Risk inference predictions by risk category.",
    ["model_name", "model_alias", "risk"],
)

# Generic classification metric used by all classification models.
#
# Unlike the legacy Vehicle Risk metric above, this metric is intentionally
# domain-agnostic so that models such as Customer Churn, Fraud Detection,
# Credit Risk, and future PyTorch classifiers can share the same observability
# contract.
ML_INFERENCE_CLASSIFICATIONS_TOTAL = Counter(
    "ml_inference_classifications_total",
    "Total ML classification predictions by predicted class.",
    ["model_name", "model_alias", "prediction_class"],
)

ML_INFERENCE_ERRORS_TOTAL = Counter(
    "ml_inference_errors_total",
    "Total ML inference errors by type.",
    ["model_name", "model_alias", "error_type"],
)
