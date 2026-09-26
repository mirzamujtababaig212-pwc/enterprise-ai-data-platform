from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.contracts import FeatureContract


OBSERVATION_WINDOW_SCHEMA_VERSION = "v1"
OBSERVATION_HISTOGRAM_BINS = 10


@dataclass(frozen=True)
class ObservationFeatureStatistics:
    """Production-observation statistics for one numeric feature."""

    count: int
    missing_count: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    quantiles: dict[str, float]
    histogram_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "missing_count": self.missing_count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "quantiles": dict(self.quantiles),
            "histogram": {
                "edges": list(self.histogram_edges),
                "counts": list(self.histogram_counts),
            },
        }


@dataclass(frozen=True)
class ObservationPredictionStatistics:
    """Production prediction statistics for one observation window."""

    count: int
    class_counts: dict[str, int]
    probability_count: int
    probability_missing_count: int
    probability_min: float | None
    probability_max: float | None
    probability_mean: float | None
    probability_std: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "class_counts": dict(self.class_counts),
            "probability": {
                "count": self.probability_count,
                "missing_count": self.probability_missing_count,
                "min": self.probability_min,
                "max": self.probability_max,
                "mean": self.probability_mean,
                "std": self.probability_std,
            },
        }


@dataclass(frozen=True)
class ObservationWindow:
    """Persistable aggregate of production model observations."""

    schema_version: str
    window_id: str
    model_name: str
    model_version: str
    model_alias: str
    training_run_id: str
    window_start: datetime
    window_end: datetime
    sample_count: int
    feature_contract_name: str
    feature_contract_version: str
    feature_names: tuple[str, ...]
    features: dict[str, ObservationFeatureStatistics]
    predictions: ObservationPredictionStatistics

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "window_id": self.window_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "training_run_id": self.training_run_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "sample_count": self.sample_count,
            "feature_contract_name": self.feature_contract_name,
            "feature_contract_version": self.feature_contract_version,
            "feature_names": list(self.feature_names),
            "features": {name: statistics.as_dict() for name, statistics in self.features.items()},
            "predictions": self.predictions.as_dict(),
        }


def build_observation_window(
    observations: pd.DataFrame,
    feature_contract: FeatureContract,
    *,
    window_id: str,
    window_start: datetime,
    window_end: datetime,
    model_name: str,
    model_version: str,
    model_alias: str,
    training_run_id: str,
) -> ObservationWindow:
    """Build a deterministic aggregate from production inference observations."""

    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")

    if observations.empty:
        raise ValueError("observations must not be empty")

    if not window_id.strip():
        raise ValueError("window_id must not be empty")

    if not model_name.strip():
        raise ValueError("model_name must not be empty")

    if not model_version.strip():
        raise ValueError("model_version must not be empty")

    if not model_alias.strip():
        raise ValueError("model_alias must not be empty")

    if not training_run_id.strip():
        raise ValueError("training_run_id must not be empty")

    if window_start.tzinfo is None or window_start.utcoffset() is None:
        raise ValueError("window_start must be timezone-aware")

    if window_end.tzinfo is None or window_end.utcoffset() is None:
        raise ValueError("window_end must be timezone-aware")

    if window_end <= window_start:
        raise ValueError("window_end must be greater than window_start")

    feature_names = feature_contract.feature_names
    missing_features = [name for name in feature_names if name not in observations.columns]

    if missing_features:
        raise ValueError(
            "observations are missing contract features: " + ", ".join(missing_features)
        )

    required_metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "training_run_id": training_run_id,
    }

    for column, expected_value in required_metadata.items():
        if column not in observations.columns:
            raise ValueError(f"observations are missing required column: {column}")

        actual_values = observations[column].dropna().astype(str).unique()

        if len(actual_values) != 1 or actual_values[0] != expected_value:
            raise ValueError(
                f"observations contain inconsistent {column}; " f"expected '{expected_value}'"
            )

    features: dict[str, ObservationFeatureStatistics] = {}

    for feature in feature_contract.features:
        series = observations[feature.name]

        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("observation window requires numeric feature: " + feature.name)

        values = pd.to_numeric(series, errors="coerce")
        valid_values = values.dropna().to_numpy(dtype=float)

        count = int(len(values))
        missing_count = int(values.isna().sum())

        if valid_values.size == 0:
            statistics = ObservationFeatureStatistics(
                count=count,
                missing_count=missing_count,
                min=None,
                max=None,
                mean=None,
                std=None,
                quantiles={},
                histogram_edges=(),
                histogram_counts=(),
            )
        else:
            minimum = float(np.min(valid_values))
            maximum = float(np.max(valid_values))

            quantile_values = np.quantile(
                valid_values,
                [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            )

            quantiles = {
                "p01": float(quantile_values[0]),
                "p05": float(quantile_values[1]),
                "p25": float(quantile_values[2]),
                "p50": float(quantile_values[3]),
                "p75": float(quantile_values[4]),
                "p95": float(quantile_values[5]),
                "p99": float(quantile_values[6]),
            }

            if minimum == maximum:
                histogram_edges = (
                    minimum - 0.5,
                    maximum + 0.5,
                )
                histogram_counts = (int(valid_values.size),)
            else:
                histogram_counts_array, histogram_edges_array = np.histogram(
                    valid_values,
                    bins=OBSERVATION_HISTOGRAM_BINS,
                    range=(minimum, maximum),
                )
                histogram_edges = tuple(float(value) for value in histogram_edges_array)
                histogram_counts = tuple(int(value) for value in histogram_counts_array)

            statistics = ObservationFeatureStatistics(
                count=count,
                missing_count=missing_count,
                min=minimum,
                max=maximum,
                mean=float(np.mean(valid_values)),
                std=float(np.std(valid_values, ddof=0)),
                quantiles=quantiles,
                histogram_edges=histogram_edges,
                histogram_counts=histogram_counts,
            )

        features[feature.name] = statistics

    if "risk" not in observations.columns:
        raise ValueError("observations are missing required column: risk")

    risk_values = observations["risk"]
    if risk_values.isna().any():
        raise ValueError("observations contain missing risk predictions")

    class_counts = {str(value): int(count) for value, count in risk_values.value_counts().items()}

    probability_column = (
        observations["risk_probability"]
        if "risk_probability" in observations.columns
        else pd.Series([np.nan] * len(observations), index=observations.index)
    )

    probability_values = pd.to_numeric(
        probability_column,
        errors="coerce",
    )
    valid_probabilities = probability_values.dropna().to_numpy(dtype=float)

    predictions = ObservationPredictionStatistics(
        count=len(observations),
        class_counts=class_counts,
        probability_count=len(probability_values),
        probability_missing_count=int(probability_values.isna().sum()),
        probability_min=(float(np.min(valid_probabilities)) if valid_probabilities.size else None),
        probability_max=(float(np.max(valid_probabilities)) if valid_probabilities.size else None),
        probability_mean=(
            float(np.mean(valid_probabilities)) if valid_probabilities.size else None
        ),
        probability_std=(
            float(np.std(valid_probabilities, ddof=0)) if valid_probabilities.size else None
        ),
    )

    return ObservationWindow(
        schema_version=OBSERVATION_WINDOW_SCHEMA_VERSION,
        window_id=window_id,
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
        training_run_id=training_run_id,
        window_start=window_start,
        window_end=window_end,
        sample_count=len(observations),
        feature_contract_name=feature_contract.name,
        feature_contract_version=feature_contract.version,
        feature_names=feature_names,
        features=features,
        predictions=predictions,
    )
