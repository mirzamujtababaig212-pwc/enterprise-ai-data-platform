from __future__ import annotations

import json

import pandas as pd
import pytest
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier

from ml.models.vehicle_risk import FEATURE_COLUMNS
from ml.sagemaker.vehicle_risk import (
    input_fn,
    model_fn,
    output_fn,
    predict_fn,
)

FEATURE_VALUES = {
    "event_count": 12,
    "avg_speed": 55.0,
    "max_speed": 72.0,
    "speed_stddev": 6.0,
    "avg_rpm": 2100.0,
    "max_rpm": 2800.0,
    "avg_fuel_level": 72.0,
    "min_fuel_level": 61.0,
    "avg_battery": 88.0,
    "avg_engine_temperature": 91.0,
    "max_engine_temperature": 98.0,
}


def _train_model():
    dataframe = pd.DataFrame(
        [
            [10, 40.0, 50.0, 2.0, 1800.0, 2100.0, 80.0, 75.0, 95.0, 85.0, 90.0],
            [11, 42.0, 52.0, 2.0, 1850.0, 2150.0, 78.0, 73.0, 94.0, 86.0, 91.0],
            [12, 70.0, 95.0, 10.0, 3000.0, 3600.0, 40.0, 25.0, 70.0, 105.0, 115.0],
            [13, 72.0, 98.0, 11.0, 3100.0, 3700.0, 38.0, 20.0, 68.0, 108.0, 118.0],
        ],
        columns=FEATURE_COLUMNS,
    )
    target = [0, 0, 1, 1]

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(dataframe, target)
    return model


def test_input_fn_accepts_vehicle_risk_json():
    dataframe = input_fn(
        json.dumps(FEATURE_VALUES),
        "application/json",
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == list(FEATURE_COLUMNS)
    assert dataframe.to_dict(orient="records") == [FEATURE_VALUES]


def test_input_fn_rejects_unknown_content_type():
    with pytest.raises(ValueError, match="Unsupported content type"):
        input_fn(
            json.dumps(FEATURE_VALUES),
            "text/plain",
        )


def test_input_fn_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        input_fn(
            "{not-json}",
            "application/json",
        )


def test_input_fn_rejects_missing_features():
    payload = dict(FEATURE_VALUES)
    payload.pop("avg_speed")

    with pytest.raises(ValueError, match="missing"):
        input_fn(
            json.dumps(payload),
            "application/json",
        )


def test_input_fn_rejects_extra_features():
    payload = dict(FEATURE_VALUES)
    payload["unexpected"] = 123

    with pytest.raises(ValueError, match="unexpected features"):
        input_fn(
            json.dumps(payload),
            "application/json",
        )


def test_model_fn_loads_skops_model(tmp_path):
    model = _train_model()
    model_path = tmp_path / "model.skops"

    sio.dump(model, str(model_path))

    loaded = model_fn(tmp_path)

    assert isinstance(loaded, RandomForestClassifier)


def test_model_fn_rejects_untrusted_types(tmp_path):
    model = _train_model()
    model_path = tmp_path / "model.skops"
    sio.dump(model, str(model_path))

    original_get_untrusted_types = sio.get_untrusted_types

    try:
        sio.get_untrusted_types = lambda **kwargs: ["evil.UntrustedType"]

        with pytest.raises(ValueError, match="untrusted types"):
            model_fn(tmp_path)
    finally:
        sio.get_untrusted_types = original_get_untrusted_types


def test_model_fn_rejects_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="model artifact not found"):
        model_fn(tmp_path)


def test_predict_fn_returns_vehicle_risk_contract():
    model = _train_model()

    dataframe = input_fn(
        json.dumps(FEATURE_VALUES),
        "application/json",
    )

    prediction = predict_fn(dataframe, model)

    assert set(prediction) == {
        "risk",
        "risk_probability",
        "model_name",
        "model_alias",
    }
    assert prediction["risk"] in (0, 1)
    assert 0.0 <= prediction["risk_probability"] <= 1.0
    assert prediction["model_name"] == "VehicleRiskModel"
    assert prediction["model_alias"] == "champion"


def test_output_fn_serializes_json_prediction():
    prediction = {
        "risk": 0,
        "risk_probability": 0.38,
        "model_name": "VehicleRiskModel",
        "model_alias": "champion",
    }

    result = output_fn(prediction, "application/json")

    assert json.loads(result) == prediction


def test_output_fn_rejects_unknown_accept_type():
    with pytest.raises(ValueError, match="Unsupported accept type"):
        output_fn(
            {"risk": 0},
            "text/plain",
        )
