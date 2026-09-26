from __future__ import annotations

import pandas as pd
import pytest

from ml.contracts import FeatureContract, FeatureDefinition
from ml.training.reference_distribution import (
    REFERENCE_HISTOGRAM_BINS,
    REFERENCE_DISTRIBUTION_SCHEMA_VERSION,
    build_reference_distribution,
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


def test_build_reference_distribution_uses_contract_order() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_b": [10, 20, 30, 40],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = build_reference_distribution(
        dataframe,
        _contract(),
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    assert result.schema_version == REFERENCE_DISTRIBUTION_SCHEMA_VERSION
    assert result.dataset_name == "test-dataset"
    assert result.dataset_version == "v1"
    assert result.feature_contract_name == "test-model"
    assert result.feature_contract_version == "v1"
    assert result.training_run_id == "run-123"
    assert result.model_name == "TestModel"
    assert result.feature_names == ("feature_a", "feature_b")
    assert list(result.features) == ["feature_a", "feature_b"]


def test_build_reference_distribution_calculates_numeric_statistics() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    result = build_reference_distribution(
        dataframe,
        _contract(),
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

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

    assert len(statistics.histogram_edges) == REFERENCE_HISTOGRAM_BINS + 1
    assert len(statistics.histogram_counts) == REFERENCE_HISTOGRAM_BINS
    assert sum(statistics.histogram_counts) == 4


def test_build_reference_distribution_tracks_missing_values() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, None, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    result = build_reference_distribution(
        dataframe,
        _contract(),
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    statistics = result.features["feature_a"]

    assert statistics.count == 4
    assert statistics.missing_count == 1
    assert statistics.min == 1.0
    assert statistics.max == 4.0
    assert statistics.mean == pytest.approx(8 / 3)
    assert sum(statistics.histogram_counts) == 3


def test_build_reference_distribution_handles_constant_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [5.0, 5.0, 5.0, 5.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    result = build_reference_distribution(
        dataframe,
        _contract(),
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    statistics = result.features["feature_a"]

    assert statistics.histogram_edges == (4.5, 5.5)
    assert statistics.histogram_counts == (4,)


def test_build_reference_distribution_rejects_missing_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing contract features: feature_b",
    ):
        build_reference_distribution(
            dataframe,
            _contract(),
            dataset_name="test-dataset",
            dataset_version="v1",
            training_run_id="run-123",
            model_name="TestModel",
        )


def test_build_reference_distribution_rejects_non_numeric_feature() -> None:
    contract = FeatureContract(
        name="test-model",
        version="v1",
        features=(
            FeatureDefinition(
                name="feature_a",
                dtype="string",
            ),
        ),
    )

    dataframe = pd.DataFrame(
        {
            "feature_a": ["a", "b", "c"],
        }
    )

    with pytest.raises(
        ValueError,
        match="requires numeric feature: feature_a",
    ):
        build_reference_distribution(
            dataframe,
            contract,
            dataset_name="test-dataset",
            dataset_version="v1",
            training_run_id="run-123",
            model_name="TestModel",
        )


def test_reference_distribution_serializes_to_json_compatible_dict() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    result = build_reference_distribution(
        dataframe,
        _contract(),
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    payload = result.as_dict()

    assert payload["schema_version"] == "v1"
    assert payload["feature_names"] == ["feature_a", "feature_b"]
    assert payload["features"]["feature_a"]["count"] == 4
    assert len(payload["features"]["feature_a"]["histogram"]["edges"]) == 11
    assert len(payload["features"]["feature_a"]["histogram"]["counts"]) == 10


def test_reference_distribution_round_trips_through_dict() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    original = build_reference_distribution(
        dataframe,
        _contract(),
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    restored = type(original).from_dict(original.as_dict())

    assert restored == original
    assert restored.features["feature_a"].histogram_edges == (
        original.features["feature_a"].histogram_edges
    )
    assert restored.features["feature_a"].histogram_counts == (
        original.features["feature_a"].histogram_counts
    )


def test_reference_distribution_from_dict_rejects_unsupported_schema() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    payload = build_reference_distribution(
        dataframe,
        _contract(),
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    ).as_dict()

    payload["schema_version"] = "v2"

    with pytest.raises(
        ValueError,
        match="unsupported reference distribution schema version",
    ):
        build_reference_distribution(
            dataframe,
            _contract(),
            dataset_name="test-dataset",
            dataset_version="v1",
            training_run_id="run-123",
            model_name="TestModel",
        ).from_dict(payload)


def test_reference_distribution_from_dict_rejects_missing_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    original = build_reference_distribution(
        dataframe,
        _contract(),
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    payload = original.as_dict()
    del payload["features"]["feature_b"]

    with pytest.raises(
        ValueError,
        match="reference feature distribution is missing features: feature_b",
    ):
        type(original).from_dict(payload)
