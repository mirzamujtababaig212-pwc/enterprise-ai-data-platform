from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml.scripts import register_vehicle_risk
from ml.platform import ModelMetadata
from ml.training import TrainingResult


def test_register_vehicle_risk_forwards_training_metadata_to_registry():
    metadata = ModelMetadata(
        model_name=register_vehicle_risk.MODEL_NAME,
        model_type="vehicle_risk_classifier",
        framework="scikit-learn",
        task_type="classification",
        target_column="risk",
        training_run_id="run-123",
        experiment_id="experiment-123",
        model_uri="runs:/run-123/model",
        lineage={
            "dataset_name": "vehicle-risk",
            "dataset_version": "v1",
            "feature_contract_name": "vehicle-risk-features",
            "feature_contract_version": "v1",
            "evaluation_policy_name": "vehicle-risk-v1",
        },
    )

    training_result = TrainingResult(
        run_id="run-123",
        experiment_id="experiment-123",
        model_uri="runs:/run-123/model",
        metrics={
            "validation_accuracy": 1.0,
            "validation_precision": 1.0,
            "validation_recall": 1.0,
            "validation_f1": 1.0,
            "validation_roc_auc": 1.0,
        },
        parameters={},
        training_samples=7,
        validation_samples=3,
        metadata=metadata,
    )

    registered = MagicMock()
    registered.model_name = register_vehicle_risk.MODEL_NAME
    registered.version = "7"
    registered.model_uri = "models:/VehicleRiskModel/7"
    registered.alias = "candidate"

    champion = MagicMock()
    champion.name = register_vehicle_risk.MODEL_NAME
    champion.version = "7"
    champion.tags = {
        "validation_status": "PASSED",
        "deployment_status": "CHAMPION",
    }

    registry = MagicMock()
    registry.register_model.return_value = registered
    registry.get_champion.return_value = champion

    quality_gate = MagicMock()
    quality_gate.passed = True
    quality_gate.errors = []

    with (
        patch("ml.scripts.register_vehicle_risk.VehicleRiskTrainer") as trainer_class,
        patch(
            "ml.scripts.register_vehicle_risk.ModelRegistryManager",
            return_value=registry,
        ),
        patch(
            "ml.scripts.register_vehicle_risk.EvaluationQualityGate.evaluate",
            return_value=quality_gate,
        ),
    ):
        trainer_class.return_value.train.return_value = training_result

        register_vehicle_risk.main()

    trainer_class.return_value.train.assert_called_once()

    registry.register_model.assert_called_once_with(
        model_uri=training_result.model_uri,
        model_name=register_vehicle_risk.MODEL_NAME,
        run_id=training_result.run_id,
        metadata=training_result.metadata,
    )

    registry.promote_to_champion.assert_called_once_with(
        model_name=register_vehicle_risk.MODEL_NAME,
        version=registered.version,
    )

    registry.get_champion.assert_called_once_with(
        register_vehicle_risk.MODEL_NAME,
    )
