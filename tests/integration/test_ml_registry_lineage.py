from unittest.mock import MagicMock

import pytest

from ml.registry import ModelRegistryManager, ModelVersionLineage

MODEL_NAME = "VehicleRiskModel"


def test_model_version_lineage_as_dict() -> None:
    lineage = ModelVersionLineage(
        model_name=MODEL_NAME,
        model_version="7",
        source_run_id="run-123",
        model_uri=f"models:/{MODEL_NAME}/7",
        dataset_name="vehicle-risk-integration",
        dataset_version="v1",
        feature_contract_name="vehicle-risk",
        feature_contract_version="v1",
        evaluation_policy_name="vehicle-risk-v1",
    )

    assert lineage.as_dict() == {
        "model_name": MODEL_NAME,
        "model_version": "7",
        "source_run_id": "run-123",
        "model_uri": f"models:/{MODEL_NAME}/7",
        "dataset_name": "vehicle-risk-integration",
        "dataset_version": "v1",
        "feature_contract_name": "vehicle-risk",
        "feature_contract_version": "v1",
        "evaluation_policy_name": "vehicle-risk-v1",
    }


def test_model_version_lineage_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="dataset_name must not be empty"):
        ModelVersionLineage(
            model_name=MODEL_NAME,
            model_version="7",
            source_run_id="run-123",
            model_uri=f"models:/{MODEL_NAME}/7",
            dataset_name="",
            dataset_version="v1",
            feature_contract_name="vehicle-risk",
            feature_contract_version="v1",
            evaluation_policy_name="vehicle-risk-v1",
        )


def test_registration_rejects_incomplete_lineage_before_model_registration() -> None:
    registry = ModelRegistryManager()

    run = MagicMock()
    run.data.tags = {
        "quality_gate_passed": "true",
        "quality_gate_policy": "vehicle-risk-v1",
    }

    registry.client.get_run = MagicMock(return_value=run)

    register_model = MagicMock()
    registry_module = __import__(
        "ml.registry.model_registry",
        fromlist=["mlflow"],
    )

    original_register_model = registry_module.mlflow.register_model
    registry_module.mlflow.register_model = register_model

    try:
        with pytest.raises(
            ValueError,
            match="missing required lineage fields",
        ):
            registry.register_model(
                model_uri="runs:/fake-run/model",
                model_name=MODEL_NAME,
                run_id="fake-run",
                metadata=MagicMock(lineage={}),
            )

        register_model.assert_not_called()
    finally:
        registry_module.mlflow.register_model = original_register_model


def test_get_model_version_lineage_reads_persisted_tags() -> None:
    registry = ModelRegistryManager()

    model_version = MagicMock()
    model_version.tags = {
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

    registry.client.get_model_version = MagicMock(return_value=model_version)

    lineage = registry.get_model_version_lineage(
        model_name=MODEL_NAME,
        version="7",
    )

    assert lineage.model_name == MODEL_NAME
    assert lineage.model_version == "7"
    assert lineage.source_run_id == "run-123"
    assert lineage.dataset_name == "vehicle-risk-integration"
    assert lineage.dataset_version == "v1"
    assert lineage.feature_contract_name == "vehicle-risk"
    assert lineage.feature_contract_version == "v1"
    assert lineage.evaluation_policy_name == "vehicle-risk-v1"


def test_get_model_version_lineage_rejects_incomplete_tags() -> None:
    registry = ModelRegistryManager()

    model_version = MagicMock()
    model_version.tags = {
        "lineage_model_name": MODEL_NAME,
        "lineage_model_version": "7",
    }

    registry.client.get_model_version = MagicMock(return_value=model_version)

    with pytest.raises(ValueError, match="incomplete lineage"):
        registry.get_model_version_lineage(
            model_name=MODEL_NAME,
            version="7",
        )
