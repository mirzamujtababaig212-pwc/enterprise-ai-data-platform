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


def test_customer_churn_model_registration_and_promotion() -> None:
    import pandas as pd

    from ml.models.customer_churn import (
        MODEL_NAME as CUSTOMER_CHURN_MODEL_NAME,
        TARGET_COLUMN as CUSTOMER_CHURN_TARGET_COLUMN,
    )
    from ml.training.customer_churn import CustomerChurnTrainer
    from ml.training.schemas import TrainingConfig

    dataframe = pd.DataFrame(
        [
            {
                "tenure_months": 6 + (index % 30),
                "monthly_charges": 50.0 + (index % 8) * 8.0,
                "total_charges": 300.0 + index * 75.0,
                "support_tickets": index % 7,
                "usage_hours": 20.0 + (index % 10) * 5.0,
                "payment_failures": index % 4,
                "churn": int(index % 3 == 0),
            }
            for index in range(40)
        ]
    )

    training_result = CustomerChurnTrainer().train(
        dataframe,
        TrainingConfig(
            experiment_name="customer-churn-registry-test",
            run_name="customer-churn-registry-test",
        ),
    )

    assert training_result.metadata is not None

    registry = ModelRegistryManager()

    registered = registry.register_model(
        model_uri=training_result.model_uri,
        model_name=CUSTOMER_CHURN_MODEL_NAME,
        run_id=training_result.run_id,
        evaluation_passed=True,
        metadata=training_result.metadata,
    )

    assert registered.model_name == CUSTOMER_CHURN_MODEL_NAME
    assert registered.alias == "candidate"

    candidate = registry.get_candidate(CUSTOMER_CHURN_MODEL_NAME)

    assert str(candidate.version) == registered.version
    assert candidate.tags.get("validation_status") == "PASSED"
    assert candidate.tags.get("model_type") == "LogisticRegression"
    assert candidate.tags.get("framework") == "scikit-learn"
    assert candidate.tags.get("task_type") == "binary_classification"
    assert candidate.tags.get("target_column") == CUSTOMER_CHURN_TARGET_COLUMN

    registry.promote_to_champion(
        model_name=CUSTOMER_CHURN_MODEL_NAME,
        version=registered.version,
    )

    champion = registry.get_champion(CUSTOMER_CHURN_MODEL_NAME)

    assert str(champion.version) == registered.version
    assert champion.tags.get("validation_status") == "PASSED"
    assert champion.tags.get("deployment_status") == "CHAMPION"
