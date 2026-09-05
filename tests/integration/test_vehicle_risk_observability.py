"""Observability tests for Vehicle Risk ML inference."""

from __future__ import annotations

import pandas as pd
import pytest
from prometheus_client import REGISTRY

from ml.inference.vehicle_risk import VehicleRiskPredictor

MODEL_NAME = "VehicleRiskModel"
MODEL_ALIAS = "champion"

VALID_FEATURES = {
    "event_count": 10,
    "avg_speed": 70.0,
    "max_speed": 80.0,
    "speed_stddev": 5.0,
    "avg_rpm": 1800.0,
    "max_rpm": 2500.0,
    "avg_fuel_level": 70.0,
    "min_fuel_level": 50.0,
    "avg_battery": 12.5,
    "avg_engine_temperature": 80.0,
    "max_engine_temperature": 90.0,
}


def get_prometheus_sample_value(
    metric_name: str,
    labels: dict[str, str],
) -> float | None:
    """
    Return the current Prometheus sample value.

    Counter samples are exposed with the `_total` suffix,
    while the registry metric family uses the base name.
    """

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == metric_name and sample.labels == labels:
                return float(sample.value)

    return None


def test_vehicle_risk_metrics_increment_on_prediction() -> None:
    """Successful inference increments ML metrics."""

    predictor = VehicleRiskPredictor(
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )

    dataframe = pd.DataFrame([VALID_FEATURES])

    request_labels = {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "status": "success",
    }

    request_before = (
        get_prometheus_sample_value(
            "ml_inference_requests_total",
            request_labels,
        )
        or 0.0
    )

    result = predictor.predict(dataframe)

    assert result.risk in (0, 1)
    assert result.model_name == MODEL_NAME
    assert result.model_alias == MODEL_ALIAS

    request_after = (
        get_prometheus_sample_value(
            "ml_inference_requests_total",
            request_labels,
        )
        or 0.0
    )

    assert request_after == request_before + 1.0

    prediction_labels = {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "risk": str(result.risk),
    }

    prediction_before = (
        get_prometheus_sample_value(
            "ml_inference_predictions_total",
            prediction_labels,
        )
        or 0.0
    )

    # The prediction was already recorded by predictor.predict().
    # Therefore the current value must be at least 1.
    assert prediction_before >= 1.0

    duration_count = get_prometheus_sample_value(
        "ml_inference_duration_seconds_count",
        {
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
        },
    )

    assert duration_count is not None
    assert duration_count >= 1.0


def test_vehicle_risk_invalid_features_record_error_metric() -> None:
    """Invalid inference input increments the ML error metric."""

    predictor = VehicleRiskPredictor(
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )

    invalid_dataframe = pd.DataFrame(
        [
            {
                "invalid_column": 123,
            }
        ]
    )

    error_labels = {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "error_type": "ValueError",
    }

    before = (
        get_prometheus_sample_value(
            "ml_inference_errors_total",
            error_labels,
        )
        or 0.0
    )

    with pytest.raises(ValueError):
        predictor.predict(invalid_dataframe)

    after = (
        get_prometheus_sample_value(
            "ml_inference_errors_total",
            error_labels,
        )
        or 0.0
    )

    assert after == before + 1.0


def test_vehicle_risk_metrics_are_registered() -> None:
    """All dedicated ML observability metric families are registered."""

    metric_names = {metric.name for metric in REGISTRY.collect()}

    assert "ml_inference_requests" in metric_names
    assert "ml_inference_duration_seconds" in metric_names
    assert "ml_inference_predictions" in metric_names
    assert "ml_inference_errors" in metric_names


def test_vehicle_risk_metric_samples_are_exposed() -> None:
    """Expected Prometheus sample names are present."""

    sample_names = {sample.name for metric in REGISTRY.collect() for sample in metric.samples}

    assert "ml_inference_requests_total" in sample_names
    assert "ml_inference_duration_seconds_count" in sample_names
    assert "ml_inference_duration_seconds_sum" in sample_names
    assert "ml_inference_predictions_total" in sample_names
    assert "ml_inference_errors_total" in sample_names
