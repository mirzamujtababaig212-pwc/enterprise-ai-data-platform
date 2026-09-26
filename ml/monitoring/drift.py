from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ml.training.reference_distribution import ReferenceFeatureDistribution
from ml.monitoring.observation_window import ObservationWindow

PSI_EPSILON = 1e-6


def calculate_population_stability_index(
    *,
    reference_counts: tuple[int, ...],
    observation_counts: tuple[int, ...],
) -> float:
    """Calculate PSI for two distributions with aligned bins."""

    if not reference_counts or not observation_counts:
        raise ValueError("distributions must contain at least one bin")

    if len(reference_counts) != len(observation_counts):
        raise ValueError(
            "reference and observation distributions must have the same number of bins"
        )

    if any(count < 0 for count in reference_counts) or any(
        count < 0 for count in observation_counts
    ):
        raise ValueError("counts must be non-negative")

    reference_total = sum(reference_counts)
    observation_total = sum(observation_counts)

    if reference_total == 0 or observation_total == 0:
        raise ValueError("distributions must contain at least one observation")

    reference_proportions = [
        max(count / reference_total, PSI_EPSILON) for count in reference_counts
    ]
    observation_proportions = [
        max(count / observation_total, PSI_EPSILON) for count in observation_counts
    ]

    reference_normalization = sum(reference_proportions)
    observation_normalization = sum(observation_proportions)

    reference_proportions = [
        proportion / reference_normalization for proportion in reference_proportions
    ]
    observation_proportions = [
        proportion / observation_normalization for proportion in observation_proportions
    ]

    return sum(
        (observation - reference) * math.log(observation / reference)
        for reference, observation in zip(
            reference_proportions,
            observation_proportions,
        )
    )


def calculate_feature_drift(
    *,
    reference_distribution,
    observation_window,
) -> dict[str, float]:
    """Calculate PSI for each feature with reference-aligned observations."""

    if reference_distribution.model_name != observation_window.model_name:
        raise ValueError("reference distribution model_name does not match observation window")

    if (
        reference_distribution.feature_contract_name != observation_window.feature_contract_name
        or reference_distribution.feature_contract_version
        != observation_window.feature_contract_version
    ):
        raise ValueError(
            "reference distribution feature contract does not match observation window"
        )

    if reference_distribution.training_run_id != observation_window.training_run_id:
        raise ValueError("reference distribution training_run_id does not match observation window")

    if reference_distribution.feature_names != observation_window.feature_names:
        raise ValueError("reference distribution features do not match observation window")

    results: dict[str, float] = {}

    for feature_name in reference_distribution.feature_names:
        reference_statistics = reference_distribution.features[feature_name]
        observation_statistics = observation_window.features[feature_name]

        if not reference_statistics.histogram_counts:
            continue

        reference_counts = (
            0,
            *reference_statistics.histogram_counts,
            0,
        )
        observation_counts = observation_statistics.reference_histogram_counts

        if not observation_counts:
            continue

        results[feature_name] = calculate_population_stability_index(
            reference_counts=reference_counts,
            observation_counts=observation_counts,
        )

    return results


DRIFT_EVALUATION_SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class DriftEvaluation:
    """Persistable feature-drift evaluation for one production observation window."""

    schema_version: str
    evaluation_id: str
    model_name: str
    model_version: str
    model_alias: str
    training_run_id: str
    observation_window_id: str
    window_start: datetime
    window_end: datetime
    feature_contract_name: str
    feature_contract_version: str
    feature_names: tuple[str, ...]
    feature_psi: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "training_run_id": self.training_run_id,
            "observation_window_id": self.observation_window_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "feature_contract_name": self.feature_contract_name,
            "feature_contract_version": self.feature_contract_version,
            "feature_names": list(self.feature_names),
            "feature_psi": dict(self.feature_psi),
        }


def build_drift_evaluation(
    *,
    reference_distribution: ReferenceFeatureDistribution,
    observation_window: ObservationWindow,
    evaluation_id: str,
) -> DriftEvaluation:
    """Build a deterministic, lineage-preserving drift evaluation."""

    if not evaluation_id.strip():
        raise ValueError("evaluation_id must not be empty")

    feature_psi = calculate_feature_drift(
        reference_distribution=reference_distribution,
        observation_window=observation_window,
    )

    return DriftEvaluation(
        schema_version=DRIFT_EVALUATION_SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        model_name=observation_window.model_name,
        model_version=observation_window.model_version,
        model_alias=observation_window.model_alias,
        training_run_id=observation_window.training_run_id,
        observation_window_id=observation_window.window_id,
        window_start=observation_window.window_start,
        window_end=observation_window.window_end,
        feature_contract_name=observation_window.feature_contract_name,
        feature_contract_version=observation_window.feature_contract_version,
        feature_names=observation_window.feature_names,
        feature_psi=feature_psi,
    )
