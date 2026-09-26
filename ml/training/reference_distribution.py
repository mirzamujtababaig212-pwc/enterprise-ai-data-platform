from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.contracts import FeatureContract


REFERENCE_DISTRIBUTION_SCHEMA_VERSION = "v1"
REFERENCE_DISTRIBUTION_ARTIFACT_PATH = "reference/reference_feature_distribution.json"
REFERENCE_HISTOGRAM_BINS = 10


@dataclass(frozen=True)
class ReferenceFeatureStatistics:
    """Training-reference statistics for one numeric feature."""

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceFeatureStatistics":
        if not isinstance(payload, dict):
            raise TypeError("reference feature statistics payload must be a dict")

        histogram = payload.get("histogram")
        if not isinstance(histogram, dict):
            raise ValueError("reference feature statistics histogram must be a dict")

        quantiles = payload.get("quantiles")
        if not isinstance(quantiles, dict):
            raise ValueError("reference feature statistics quantiles must be a dict")

        edges = histogram.get("edges")
        counts = histogram.get("counts")
        if not isinstance(edges, list) or not isinstance(counts, list):
            raise ValueError("reference feature statistics histogram must contain lists")

        return cls(
            count=int(payload["count"]),
            missing_count=int(payload["missing_count"]),
            min=None if payload["min"] is None else float(payload["min"]),
            max=None if payload["max"] is None else float(payload["max"]),
            mean=None if payload["mean"] is None else float(payload["mean"]),
            std=None if payload["std"] is None else float(payload["std"]),
            quantiles={name: float(value) for name, value in quantiles.items()},
            histogram_edges=tuple(float(value) for value in edges),
            histogram_counts=tuple(int(value) for value in counts),
        )


@dataclass(frozen=True)
class ReferenceFeatureDistribution:
    """Persistable training-reference distribution for a model."""

    schema_version: str
    dataset_name: str
    dataset_version: str
    feature_contract_name: str
    feature_contract_version: str
    training_run_id: str
    model_name: str
    feature_names: tuple[str, ...]
    features: dict[str, ReferenceFeatureStatistics]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "feature_contract_name": self.feature_contract_name,
            "feature_contract_version": self.feature_contract_version,
            "training_run_id": self.training_run_id,
            "model_name": self.model_name,
            "feature_names": list(self.feature_names),
            "features": {name: statistics.as_dict() for name, statistics in self.features.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceFeatureDistribution":
        if not isinstance(payload, dict):
            raise TypeError("reference feature distribution payload must be a dict")

        schema_version = payload.get("schema_version")
        if schema_version != REFERENCE_DISTRIBUTION_SCHEMA_VERSION:
            raise ValueError(
                "unsupported reference distribution schema version: " f"{schema_version!r}"
            )

        feature_names = payload.get("feature_names")
        features = payload.get("features")

        if not isinstance(feature_names, list):
            raise ValueError("reference feature distribution feature_names must be a list")

        if not isinstance(features, dict):
            raise ValueError("reference feature distribution features must be a dict")

        missing_features = [name for name in feature_names if name not in features]
        if missing_features:
            raise ValueError(
                "reference feature distribution is missing features: " + ", ".join(missing_features)
            )

        return cls(
            schema_version=schema_version,
            dataset_name=str(payload["dataset_name"]),
            dataset_version=str(payload["dataset_version"]),
            feature_contract_name=str(payload["feature_contract_name"]),
            feature_contract_version=str(payload["feature_contract_version"]),
            training_run_id=str(payload["training_run_id"]),
            model_name=str(payload["model_name"]),
            feature_names=tuple(feature_names),
            features={
                name: ReferenceFeatureStatistics.from_dict(features[name]) for name in feature_names
            },
        )


def load_reference_distribution(
    artifact_path: str | Path,
) -> ReferenceFeatureDistribution:
    """Load a persisted reference distribution JSON artifact."""

    artifact_path = Path(artifact_path)

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Reference distribution artifact not found: {artifact_path}")

    with artifact_path.open(encoding="utf-8") as artifact_file:
        payload = json.load(artifact_file)

    return ReferenceFeatureDistribution.from_dict(payload)


def build_reference_distribution(
    X_train: pd.DataFrame,
    feature_contract: FeatureContract,
    *,
    dataset_name: str,
    dataset_version: str,
    training_run_id: str,
    model_name: str,
) -> ReferenceFeatureDistribution:
    """Build a deterministic reference distribution from the exact X_train."""

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")

    if X_train.empty:
        raise ValueError("X_train must not be empty")

    feature_names = feature_contract.feature_names
    missing_features = [name for name in feature_names if name not in X_train.columns]

    if missing_features:
        raise ValueError("X_train is missing contract features: " + ", ".join(missing_features))

    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty")

    if not dataset_version.strip():
        raise ValueError("dataset_version must not be empty")

    if not training_run_id.strip():
        raise ValueError("training_run_id must not be empty")

    if not model_name.strip():
        raise ValueError("model_name must not be empty")

    features: dict[str, ReferenceFeatureStatistics] = {}

    for feature in feature_contract.features:
        series = X_train[feature.name]

        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"reference distribution requires numeric feature: " f"{feature.name}")

        values = pd.to_numeric(series, errors="coerce")
        valid_values = values.dropna().to_numpy(dtype=float)

        count = int(len(values))
        missing_count = int(values.isna().sum())

        if valid_values.size == 0:
            statistics = ReferenceFeatureStatistics(
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
                    bins=REFERENCE_HISTOGRAM_BINS,
                    range=(minimum, maximum),
                )
                histogram_edges = tuple(float(value) for value in histogram_edges_array)
                histogram_counts = tuple(int(value) for value in histogram_counts_array)

            statistics = ReferenceFeatureStatistics(
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

    return ReferenceFeatureDistribution(
        schema_version=REFERENCE_DISTRIBUTION_SCHEMA_VERSION,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        feature_contract_name=feature_contract.name,
        feature_contract_version=feature_contract.version,
        training_run_id=training_run_id,
        model_name=model_name,
        feature_names=feature_names,
        features=features,
    )
