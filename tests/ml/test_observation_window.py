from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from ml.contracts import FeatureContract, FeatureDefinition
from ml.monitoring.observation_window import (
    OBSERVATION_HISTOGRAM_BINS,
    OBSERVATION_WINDOW_SCHEMA_VERSION,
    build_observation_window,
)


def _contract() -> FeatureContract:
    return FeatureContract(
        name="test-model",
        version="v1",
        features=(
            FeatureDefinition(
                name="feature_a",
                dtype="float",
            ),
            FeatureDefinition(
                name="feature_b",
                dtype="int",
            ),
        ),
    )


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
            "risk": [0, 0, 1, 1],
            "risk_probability": [0.1, 0.2, 0.8, 0.9],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )


def _window() -> object:
    return build_observation_window(
        _observations(),
        _contract(),
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
    )


def test_build_observation_window_preserves_identity() -> None:
    result = _window()

    assert result.schema_version == OBSERVATION_WINDOW_SCHEMA_VERSION
    assert result.window_id == "window-001"
    assert result.model_name == "TestModel"
    assert result.model_version == "2"
    assert result.model_alias == "champion"
    assert result.training_run_id == "run-123"
    assert result.window_start == datetime(2026, 9, 25, tzinfo=UTC)
    assert result.window_end == datetime(2026, 9, 26, tzinfo=UTC)
    assert result.sample_count == 4
    assert result.feature_contract_name == "test-model"
    assert result.feature_contract_version == "v1"
    assert result.feature_names == ("feature_a", "feature_b")


def test_build_observation_window_calculates_feature_statistics() -> None:
    result = _window()
    statistics = result.features["feature_a"]

    assert statistics.count == 4
    assert statistics.missing_count == 0
    assert statistics.min == 1.0
    assert statistics.max == 4.0
    assert statistics.mean == 2.5
    assert statistics.std == pytest.approx(1.1180339887)

    assert statistics.quantiles["p01"] == pytest.approx(1.03)
    assert statistics.quantiles["p05"] == pytest.approx(1.15)
    assert statistics.quantiles["p25"] == pytest.approx(1.75)
    assert statistics.quantiles["p50"] == pytest.approx(2.5)
    assert statistics.quantiles["p75"] == pytest.approx(3.25)
    assert statistics.quantiles["p95"] == pytest.approx(3.85)
    assert statistics.quantiles["p99"] == pytest.approx(3.97)

    assert len(statistics.histogram_edges) == OBSERVATION_HISTOGRAM_BINS + 1
    assert len(statistics.histogram_counts) == OBSERVATION_HISTOGRAM_BINS
    assert sum(statistics.histogram_counts) == 4


def test_build_observation_window_calculates_prediction_statistics() -> None:
    result = _window()

    assert result.predictions.count == 4
    assert result.predictions.class_counts == {
        "0": 2,
        "1": 2,
    }
    assert result.predictions.probability_count == 4
    assert result.predictions.probability_missing_count == 0
    assert result.predictions.probability_min == 0.1
    assert result.predictions.probability_max == 0.9
    assert result.predictions.probability_mean == pytest.approx(0.5)
    assert result.predictions.probability_std == pytest.approx(0.3535533906)


def test_build_observation_window_tracks_missing_feature_values() -> None:
    observations = _observations()
    observations.loc[1, "feature_a"] = None

    result = build_observation_window(
        observations,
        _contract(),
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
    )

    statistics = result.features["feature_a"]

    assert statistics.count == 4
    assert statistics.missing_count == 1
    assert statistics.mean == pytest.approx(8 / 3)
    assert sum(statistics.histogram_counts) == 3


def test_build_observation_window_rejects_mixed_model_versions() -> None:
    observations = _observations()
    observations.loc[1, "model_version"] = "3"

    with pytest.raises(
        ValueError,
        match="inconsistent model_version",
    ):
        _window_with(observations)


def test_build_observation_window_rejects_missing_feature() -> None:
    observations = _observations().drop(columns=["feature_b"])

    with pytest.raises(
        ValueError,
        match="missing contract features: feature_b",
    ):
        _window_with(observations)


def test_build_observation_window_rejects_naive_window_timestamps() -> None:
    with pytest.raises(
        ValueError,
        match="window_start must be timezone-aware",
    ):
        build_observation_window(
            _observations(),
            _contract(),
            window_id="window-001",
            window_start=datetime(2026, 9, 25),
            window_end=datetime(2026, 9, 26, tzinfo=UTC),
            model_name="TestModel",
            model_version="2",
            model_alias="champion",
            training_run_id="run-123",
        )


def test_build_observation_window_rejects_invalid_window_range() -> None:
    with pytest.raises(
        ValueError,
        match="window_end must be greater than window_start",
    ):
        build_observation_window(
            _observations(),
            _contract(),
            window_id="window-001",
            window_start=datetime(2026, 9, 26, tzinfo=UTC),
            window_end=datetime(2026, 9, 25, tzinfo=UTC),
            model_name="TestModel",
            model_version="2",
            model_alias="champion",
            training_run_id="run-123",
        )


def test_observation_window_serializes_to_json_compatible_dict() -> None:
    result = _window()

    payload = result.as_dict()

    assert payload["schema_version"] == "v1"
    assert payload["window_id"] == "window-001"
    assert payload["window_start"] == "2026-09-25T00:00:00+00:00"
    assert payload["window_end"] == "2026-09-26T00:00:00+00:00"
    assert payload["feature_names"] == ["feature_a", "feature_b"]
    assert payload["features"]["feature_a"]["count"] == 4
    assert len(payload["features"]["feature_a"]["histogram"]["edges"]) == 11
    assert len(payload["features"]["feature_a"]["histogram"]["counts"]) == 10
    assert payload["predictions"]["class_counts"] == {
        "0": 2,
        "1": 2,
    }


def test_build_observation_window_rejects_missing_risk_prediction() -> None:
    observations = _observations().drop(columns=["risk"])

    with pytest.raises(
        ValueError,
        match="missing required column: risk",
    ):
        _window_with(observations)


def test_build_observation_window_rejects_missing_risk_values() -> None:
    observations = _observations()
    observations.loc[0, "risk"] = None

    with pytest.raises(
        ValueError,
        match="missing risk predictions",
    ):
        _window_with(observations)


def _window_with(observations: pd.DataFrame):
    return build_observation_window(
        observations,
        _contract(),
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
    )
