from __future__ import annotations

import mlflow
import pytest
import os
from ml.registry import ModelRegistryManager
from unittest.mock import MagicMock

MODEL_NAME = "VehicleRiskModel"


@pytest.fixture
def registry() -> ModelRegistryManager:
    return ModelRegistryManager()


def test_registration_rejected_when_persisted_evaluation_fails(
    registry: ModelRegistryManager,
) -> None:
    run = MagicMock()
    run.data.tags = {
        "quality_gate_passed": "false",
        "quality_gate_policy": "vehicle-risk-v1",
        "quality_gate_errors": ('["validation_f1=0.7000 is below required minimum 0.9000"]'),
    }

    registry.client.get_run = MagicMock(return_value=run)

    with pytest.raises(ValueError, match="quality gate failed"):
        registry.register_model(
            model_uri="runs:/fake-run/model",
            model_name="VehicleRiskModel",
            run_id="fake-run",
        )


def test_registration_rejected_when_quality_gate_is_missing(
    registry: ModelRegistryManager,
) -> None:
    run = MagicMock()
    run.data.tags = {}

    registry.client.get_run = MagicMock(return_value=run)

    with pytest.raises(ValueError, match="no persisted quality-gate decision"):
        registry.register_model(
            model_uri="runs:/fake-run/model",
            model_name="VehicleRiskModel",
            run_id="fake-run",
        )


def test_registration_rejected_when_quality_gate_policy_mismatches(
    registry: ModelRegistryManager,
) -> None:
    run = MagicMock()
    run.data.tags = {
        "quality_gate_passed": "true",
        "quality_gate_policy": "customer-churn-v1",
        "quality_gate_errors": "[]",
    }

    registry.client.get_run = MagicMock(return_value=run)

    with pytest.raises(ValueError, match="requires policy"):
        registry.register_model(
            model_uri="runs:/fake-run/model",
            model_name="VehicleRiskModel",
            run_id="fake-run",
        )


def test_registration_rejected_when_quality_gate_is_malformed(
    registry: ModelRegistryManager,
) -> None:
    run = MagicMock()
    run.data.tags = {
        "quality_gate_passed": "yes",
        "quality_gate_policy": "vehicle-risk-v1",
    }

    registry.client.get_run = MagicMock(return_value=run)

    with pytest.raises(ValueError, match="invalid quality_gate_passed"):
        registry.register_model(
            model_uri="runs:/fake-run/model",
            model_name="VehicleRiskModel",
            run_id="fake-run",
        )


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


def test_promotion_rejected_when_model_version_lineage_is_incomplete(
    registry: ModelRegistryManager,
) -> None:
    target = MagicMock()
    target.version = "7"
    target.tags = {
        "validation_status": "PASSED",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    with pytest.raises(ValueError, match="incomplete lineage"):
        registry.promote_to_champion(
            model_name=MODEL_NAME,
            version="7",
        )


def test_promotion_rejected_when_lineage_model_identity_mismatches(
    registry: ModelRegistryManager,
) -> None:
    target = MagicMock()
    target.version = "7"
    target.tags = {
        "validation_status": "PASSED",
        "lineage_model_name": "DifferentModel",
        "lineage_model_version": "7",
        "lineage_source_run_id": "run-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/7",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "vehicle-risk-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    with pytest.raises(ValueError, match="lineage model name"):
        registry.promote_to_champion(
            model_name=MODEL_NAME,
            version="7",
        )


def test_promotion_rejected_when_lineage_policy_mismatches(
    registry: ModelRegistryManager,
) -> None:
    target = MagicMock()
    target.version = "7"
    target.tags = {
        "validation_status": "PASSED",
        "lineage_model_name": MODEL_NAME,
        "source_run_id": "run-123",
        "lineage_model_version": "7",
        "lineage_source_run_id": "run-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/7",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "customer-churn-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    with pytest.raises(ValueError, match="lineage evaluation policy"):
        registry.promote_to_champion(
            model_name=MODEL_NAME,
            version="7",
        )


def test_promotion_rejected_when_source_quality_gate_fails(
    registry: ModelRegistryManager,
) -> None:
    target = MagicMock()
    target.version = "7"
    target.tags = {
        "validation_status": "PASSED",
        "source_run_id": "run-123",
        "lineage_model_name": MODEL_NAME,
        "lineage_model_version": "7",
        "lineage_source_run_id": "run-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/7",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "vehicle-risk-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    run = MagicMock()
    run.data.tags = {
        "quality_gate_passed": "false",
        "quality_gate_policy": "vehicle-risk-v1",
        "quality_gate_errors": "validation_f1 below threshold",
    }

    registry.client.get_run = MagicMock(return_value=run)

    with pytest.raises(ValueError, match="quality gate failed"):
        registry.promote_to_champion(
            model_name=MODEL_NAME,
            version="7",
        )


def test_promotion_rejected_when_lineage_source_run_mismatches(
    registry: ModelRegistryManager,
) -> None:
    target = MagicMock()
    target.version = "7"
    target.tags = {
        "validation_status": "PASSED",
        "source_run_id": "run-456",
        "lineage_model_name": MODEL_NAME,
        "lineage_model_version": "7",
        "lineage_source_run_id": "run-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/7",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "vehicle-risk-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    with pytest.raises(ValueError, match="lineage source run ID"):
        registry.promote_to_champion(
            model_name=MODEL_NAME,
            version="7",
        )


def test_failed_promotion_does_not_retire_existing_champion(
    registry: ModelRegistryManager,
) -> None:
    current_champion = MagicMock()
    current_champion.version = "5"
    current_champion.tags = {
        "deployment_status": "CHAMPION",
        "validation_status": "PASSED",
    }

    target = MagicMock()
    target.version = "7"
    target.tags = {
        "validation_status": "PASSED",
        "source_run_id": "run-123",
        "lineage_model_name": MODEL_NAME,
        "lineage_model_version": "7",
        "lineage_source_run_id": "run-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/7",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "customer-churn-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)
    registry.client.set_model_version_tag = MagicMock()
    registry.client.set_registered_model_alias = MagicMock()

    registry._get_current_champion = MagicMock(
        return_value=current_champion,
    )

    with pytest.raises(ValueError, match="lineage evaluation policy"):
        registry.promote_to_champion(
            model_name=MODEL_NAME,
            version="7",
        )

    assert current_champion.tags["deployment_status"] == "CHAMPION"

    registry.client.set_model_version_tag.assert_not_called()
    registry.client.set_registered_model_alias.assert_not_called()


def test_rollback_rejected_when_model_version_lineage_is_incomplete(
    registry: ModelRegistryManager,
) -> None:
    target = MagicMock()
    target.version = "5"
    target.tags = {
        "validation_status": "PASSED",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    with pytest.raises(ValueError, match="incomplete lineage"):
        registry.rollback_to_version(
            model_name=MODEL_NAME,
            version="5",
        )


def test_rollback_rejected_when_source_quality_gate_fails(
    registry: ModelRegistryManager,
) -> None:
    target = MagicMock()
    target.version = "5"
    target.tags = {
        "validation_status": "PASSED",
        "source_run_id": "run-rollback-123",
        "lineage_model_name": MODEL_NAME,
        "lineage_model_version": "5",
        "lineage_source_run_id": "run-rollback-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/5",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "vehicle-risk-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    run = MagicMock()
    run.data.tags = {
        "quality_gate_passed": "false",
        "quality_gate_policy": "vehicle-risk-v1",
        "quality_gate_errors": "validation_f1 below threshold",
    }

    registry.client.get_run = MagicMock(return_value=run)

    with pytest.raises(ValueError, match="quality gate failed"):
        registry.rollback_to_version(
            model_name=MODEL_NAME,
            version="5",
        )


def test_rollback_rejected_when_lineage_policy_mismatches(
    registry: ModelRegistryManager,
) -> None:
    target = MagicMock()
    target.version = "5"
    target.tags = {
        "validation_status": "PASSED",
        "source_run_id": "run-rollback-123",
        "lineage_model_name": MODEL_NAME,
        "lineage_model_version": "5",
        "lineage_source_run_id": "run-rollback-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/5",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "customer-churn-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    with pytest.raises(ValueError, match="lineage evaluation policy"):
        registry.rollback_to_version(
            model_name=MODEL_NAME,
            version="5",
        )


def test_failed_rollback_does_not_retire_existing_champion(
    registry: ModelRegistryManager,
) -> None:
    current_champion = MagicMock()
    current_champion.version = "7"
    current_champion.tags = {
        "deployment_status": "CHAMPION",
        "validation_status": "PASSED",
    }

    target = MagicMock()
    target.version = "5"
    target.tags = {
        "validation_status": "PASSED",
        "source_run_id": "run-rollback-123",
        "lineage_model_name": MODEL_NAME,
        "lineage_model_version": "5",
        "lineage_source_run_id": "run-rollback-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/5",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "customer-churn-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)
    registry.client.set_model_version_tag = MagicMock()
    registry.client.set_registered_model_alias = MagicMock()

    registry._get_current_champion = MagicMock(
        return_value=current_champion,
    )

    with pytest.raises(ValueError, match="lineage evaluation policy"):
        registry.rollback_to_version(
            model_name=MODEL_NAME,
            version="5",
        )

    assert current_champion.tags["deployment_status"] == "CHAMPION"

    registry.client.set_model_version_tag.assert_not_called()
    registry.client.set_registered_model_alias.assert_not_called()


def test_valid_historical_version_can_be_rolled_back_to_champion(
    registry: ModelRegistryManager,
) -> None:
    current_champion = MagicMock()
    current_champion.version = "7"
    current_champion.tags = {
        "deployment_status": "CHAMPION",
        "validation_status": "PASSED",
    }

    target = MagicMock()
    target.version = "5"
    target.tags = {
        "validation_status": "PASSED",
        "source_run_id": "run-rollback-123",
        "lineage_model_name": MODEL_NAME,
        "lineage_model_version": "5",
        "lineage_source_run_id": "run-rollback-123",
        "lineage_model_uri": f"models:/{MODEL_NAME}/5",
        "lineage_dataset_name": "vehicle-risk-integration",
        "lineage_dataset_version": "v1",
        "lineage_feature_contract_name": "vehicle-risk",
        "lineage_feature_contract_version": "v1",
        "lineage_evaluation_policy_name": "vehicle-risk-v1",
    }

    registry.client.get_model_version = MagicMock(return_value=target)

    run = MagicMock()
    run.data.tags = {
        "quality_gate_passed": "true",
        "quality_gate_policy": "vehicle-risk-v1",
    }

    registry.client.get_run = MagicMock(return_value=run)

    registry._get_current_champion = MagicMock(
        return_value=current_champion,
    )

    registry.client.set_model_version_tag = MagicMock()
    registry.client.set_registered_model_alias = MagicMock()

    registry.rollback_to_version(
        model_name=MODEL_NAME,
        version="5",
    )

    registry.client.set_model_version_tag.assert_any_call(
        name=MODEL_NAME,
        version="7",
        key="deployment_status",
        value="RETIRED",
    )

    registry.client.set_model_version_tag.assert_any_call(
        name=MODEL_NAME,
        version="5",
        key="deployment_status",
        value="CHAMPION",
    )

    registry.client.set_registered_model_alias.assert_called_once_with(
        name=MODEL_NAME,
        alias="champion",
        version="5",
    )
