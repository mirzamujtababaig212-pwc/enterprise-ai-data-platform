from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from ai_platform.mlflow.client import MLflowManager
from ml.contracts import FeatureContract, FeatureDefinition
from ml.monitoring import (
    ObservationWindow,
    build_observation_window,
)
from ml.monitoring.persistence import (
    MONITORING_EXPERIMENT_NAME,
    observation_artifact_path,
    persist_observation_window,
)


def _contract() -> FeatureContract:
    return FeatureContract(
        name="integration-monitoring-contract",
        version="v1",
        features=(
            FeatureDefinition(name="feature_a", dtype="float64"),
            FeatureDefinition(name="feature_b", dtype="int64"),
        ),
    )


def _window() -> ObservationWindow:
    observations = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
            "risk": [0, 0, 1, 1],
            "risk_probability": [0.1, 0.2, 0.8, 0.9],
            "model_name": ["IntegrationModel"] * 4,
            "model_version": ["7"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["training-run-123"] * 4,
        }
    )

    return build_observation_window(
        observations,
        _contract(),
        window_id="integration-window-001",
        model_name="IntegrationModel",
        model_version="7",
        model_alias="champion",
        training_run_id="training-run-123",
        window_start=datetime(2026, 9, 25, tzinfo=timezone.utc),
        window_end=datetime(2026, 9, 26, tzinfo=timezone.utc),
    )


def test_persist_observation_window_end_to_end() -> None:
    window = _window()

    manager = MLflowManager()

    run_id = persist_observation_window(
        window,
        mlflow_manager=manager,
    )

    assert run_id

    run = manager.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.info.run_name == "observation-window-integration-window-001"

    assert run.data.tags["monitoring_type"] == "observation_window"
    assert run.data.tags["observation_window_id"] == window.window_id
    assert run.data.tags["model_name"] == window.model_name
    assert run.data.tags["model_version"] == window.model_version
    assert run.data.tags["model_alias"] == window.model_alias
    assert run.data.tags["training_run_id"] == window.training_run_id
    assert run.data.tags["feature_contract_name"] == window.feature_contract_name
    assert run.data.tags["feature_contract_version"] == window.feature_contract_version
    assert run.data.tags["window_start"] == window.window_start.isoformat()
    assert run.data.tags["window_end"] == window.window_end.isoformat()

    experiment = manager.client.get_experiment(run.info.experiment_id)

    assert experiment is not None
    assert experiment.name == MONITORING_EXPERIMENT_NAME

    artifact_path = manager.client.download_artifacts(
        run_id,
        observation_artifact_path(window),
    )

    with open(artifact_path, encoding="utf-8") as artifact_file:
        persisted_window = json.load(artifact_file)

    assert persisted_window == window.as_dict()
