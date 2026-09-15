from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import skops.io as sio

from ml.models.vehicle_risk import (
    FEATURE_COLUMNS,
    MODEL_NAME,
    validate_feature_dataframe,
)

MODEL_FILENAME = "model.skops"
DEFAULT_MODEL_ALIAS = "champion"
JSON_CONTENT_TYPE = "application/json"


def model_fn(model_dir: str | Path) -> Any:
    """Load the packaged Vehicle Risk model from the SageMaker model directory."""
    model_path = Path(model_dir) / MODEL_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(f"Vehicle Risk model artifact not found: {model_path}")

    untrusted_types = sio.get_untrusted_types(file=str(model_path))

    if untrusted_types:
        raise ValueError(
            "Vehicle Risk model artifact contains untrusted types: " + ", ".join(untrusted_types)
        )

    return sio.load(
        str(model_path),
        trusted=[],
    )


def input_fn(request_body: str | bytes, content_type: str) -> pd.DataFrame:
    """Deserialize a JSON Vehicle Risk request into a validated DataFrame."""
    if content_type != JSON_CONTENT_TYPE:
        raise ValueError(f"Unsupported content type: {content_type}")

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    try:
        payload = json.loads(request_body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Request body must contain valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Vehicle Risk request must be a JSON object")

    dataframe = pd.DataFrame([payload])

    expected_columns = set(FEATURE_COLUMNS)
    actual_columns = set(dataframe.columns)

    missing = sorted(expected_columns - actual_columns)
    extra = sorted(actual_columns - expected_columns)

    if missing:
        raise ValueError(f"Vehicle Risk request is missing features: {missing}")

    if extra:
        raise ValueError(f"Vehicle Risk request contains unexpected features: {extra}")

    validate_feature_dataframe(dataframe)

    return dataframe[list(FEATURE_COLUMNS)]


def predict_fn(
    input_data: pd.DataFrame,
    model: Any,
) -> dict[str, Any]:
    """Execute Vehicle Risk inference against the packaged model."""
    validate_feature_dataframe(input_data)

    selected_features = input_data[list(FEATURE_COLUMNS)]

    prediction = model.predict(selected_features)
    risk = int(prediction[0])

    probability: float | None = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(selected_features)

        if probabilities.shape[1] >= 2:
            probability = float(probabilities[0][1])

    return {
        "risk": risk,
        "risk_probability": probability,
        "model_name": MODEL_NAME,
        "model_alias": DEFAULT_MODEL_ALIAS,
    }


def output_fn(prediction: dict[str, Any], accept: str) -> str:
    """Serialize the Vehicle Risk prediction as JSON."""
    if accept != JSON_CONTENT_TYPE:
        raise ValueError(f"Unsupported accept type: {accept}")

    return json.dumps(prediction)
