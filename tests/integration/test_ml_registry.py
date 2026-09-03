from __future__ import annotations

import mlflow
import pytest
import os
from ml.registry import ModelRegistryManager


MODEL_NAME = "VehicleRiskModel"


@pytest.fixture
def registry() -> ModelRegistryManager:
    return ModelRegistryManager()


def test_vehicle_risk_champion_exists(
    registry: ModelRegistryManager,
) -> None:
    champion = registry.get_champion(MODEL_NAME)

    assert champion.name == MODEL_NAME

    assert champion.version is not None

    assert champion.tags.get("validation_status") == "PASSED"

    assert champion.tags.get("deployment_status") == "CHAMPION"


def test_vehicle_risk_versions_exist(
    registry: ModelRegistryManager,
) -> None:
    versions = registry.list_versions(MODEL_NAME)

    assert len(versions) >= 1

    for version in versions:
        assert version.version is not None
        assert version.status == "READY"


def test_exactly_one_champion(
    registry: ModelRegistryManager,
) -> None:

    versions = registry.list_versions(MODEL_NAME)

    champions = [
        version for version in versions if version.tags.get("deployment_status") == "CHAMPION"
    ]

    assert len(champions) == 1

    active_champion = registry.get_champion(MODEL_NAME)

    assert str(active_champion.version) == str(champions[0].version)


def test_candidate_exists(
    registry: ModelRegistryManager,
) -> None:

    candidate = registry.get_candidate(MODEL_NAME)

    assert candidate.name == MODEL_NAME

    assert candidate.version is not None

    assert candidate.tags.get("validation_status") == "PASSED"


def test_champion_model_uri_loads() -> None:

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://mlflow:5000",
        )
    )

    uri = f"models:/{MODEL_NAME}@champion"

    model = mlflow.sklearn.load_model(uri)

    assert model is not None

    assert type(model).__name__ == ("RandomForestClassifier")


def test_champion_uri(
    registry: ModelRegistryManager,
) -> None:

    uri = registry.get_champion_uri(MODEL_NAME)

    assert uri == (f"models:/{MODEL_NAME}@champion")
