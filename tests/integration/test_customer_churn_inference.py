from __future__ import annotations

import pandas as pd
import pytest
from prometheus_client import REGISTRY

from ml.inference.customer_churn import CustomerChurnPredictor

MODEL_NAME = "CustomerChurnModel"
MODEL_ALIAS = "champion"

VALID_FEATURES = {
    "tenure_months": 8,
    "monthly_charges": 95.0,
    "total_charges": 760.0,
    "support_tickets": 6,
    "usage_hours": 120.0,
    "payment_failures": 3,
}


def get_prometheus_sample_value(
    metric_name: str,
    labels: dict[str, str],
) -> float | None:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == metric_name and sample.labels == labels:
                return float(sample.value)

    return None


def test_customer_churn_champion_inference() -> None:
    predictor = CustomerChurnPredictor(
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )

    dataframe = pd.DataFrame([VALID_FEATURES])

    result = predictor.predict(dataframe)

    assert result.churn in (0, 1)
    assert result.model_name == MODEL_NAME
    assert result.model_alias == MODEL_ALIAS

    assert result.churn_probability is not None
    assert 0.0 <= result.churn_probability <= 1.0


def test_customer_churn_batch_inference() -> None:
    predictor = CustomerChurnPredictor(
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )

    dataframe = pd.DataFrame(
        [
            VALID_FEATURES,
            {
                **VALID_FEATURES,
                "tenure_months": 48,
                "monthly_charges": 45.0,
                "support_tickets": 1,
                "payment_failures": 0,
            },
        ]
    )

    result = predictor.predict_batch(dataframe)

    assert len(result) == 2
    assert "churn" in result.columns
    assert "churn_probability" in result.columns
    assert "model_name" in result.columns
    assert "model_alias" in result.columns

    assert set(result["model_name"]) == {MODEL_NAME}
    assert set(result["model_alias"]) == {MODEL_ALIAS}
    assert result["churn"].isin([0, 1]).all()


def test_customer_churn_missing_feature_rejected() -> None:
    predictor = CustomerChurnPredictor(
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )

    dataframe = pd.DataFrame(
        [
            {
                "tenure_months": 8,
                "monthly_charges": 95.0,
            }
        ]
    )

    with pytest.raises(ValueError):
        predictor.predict(dataframe)


def test_customer_churn_classification_metric_incremented() -> None:
    predictor = CustomerChurnPredictor(
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )

    dataframe = pd.DataFrame([VALID_FEATURES])

    result = predictor.predict(dataframe)

    labels = {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "prediction_class": str(result.churn),
    }

    value = get_prometheus_sample_value(
        "ml_inference_classifications_total",
        labels,
    )

    assert value is not None
    assert value >= 1.0
