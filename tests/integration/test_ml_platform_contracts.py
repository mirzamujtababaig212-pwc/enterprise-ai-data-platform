from __future__ import annotations

import pytest

from ml.platform import (
    EvaluationService,
    InferenceService,
    ModelMetadata,
    ModelRegistry,
    TrainingService,
)


def test_model_metadata_is_immutable_and_copies_mutable_fields() -> None:
    metrics = {"accuracy": 0.95}
    parameters = {"depth": 5}
    tags = {"environment": "test"}

    metadata = ModelMetadata(
        model_name="VehicleRiskModel",
        model_version="1",
        model_type="classifier",
        framework="scikit-learn",
        task_type="binary_classification",
        metrics=metrics,
        parameters=parameters,
        tags=tags,
    )

    metrics["accuracy"] = 0.10
    parameters["depth"] = 99
    tags["environment"] = "production"

    assert metadata.metrics == {"accuracy": 0.95}
    assert metadata.parameters == {"depth": 5}
    assert metadata.tags == {"environment": "test"}

    with pytest.raises(AttributeError):
        metadata.model_name = "changed"


def test_model_metadata_requires_model_name() -> None:
    with pytest.raises(ValueError, match="model_name must not be empty"):
        ModelMetadata(model_name="")


def test_training_service_is_abstract() -> None:
    with pytest.raises(TypeError):
        TrainingService()


def test_evaluation_service_is_abstract() -> None:
    with pytest.raises(TypeError):
        EvaluationService()


def test_inference_service_is_abstract() -> None:
    with pytest.raises(TypeError):
        InferenceService()


def test_model_registry_is_abstract() -> None:
    with pytest.raises(TypeError):
        ModelRegistry()


def test_model_metadata_supports_framework_agnostic_models() -> None:
    metadata = ModelMetadata(
        model_name="customer-churn",
        model_version="7",
        model_type="classifier",
        framework="pytorch",
        task_type="binary_classification",
        feature_names=("tenure", "monthly_charges"),
    )

    assert metadata.model_name == "customer-churn"
    assert metadata.framework == "pytorch"
    assert metadata.feature_names == (
        "tenure",
        "monthly_charges",
    )


def test_evaluation_service_supports_framework_agnostic_output() -> None:
    class Evaluation:
        def evaluate(self, evaluation_input):
            return {"accuracy": 0.95}

    class ConcreteEvaluator(EvaluationService[object, dict[str, float]]):
        def evaluate(self, evaluation_input):
            return {"accuracy": 0.95}

    evaluator = ConcreteEvaluator()

    assert evaluator.evaluate(Evaluation()) == {"accuracy": 0.95}


def test_model_registry_supports_framework_agnostic_version_output() -> None:
    class ConcreteRegistry(ModelRegistry[dict[str, str]]):
        def register_model(
            self,
            model_uri,
            model_name,
            run_id,
            evaluation_passed,
        ):
            return {
                "model_name": model_name,
                "version": "1",
            }

        def promote_to_champion(self, model_name, version):
            return None

        def rollback_to_version(self, model_name, version):
            return None

        def get_champion_uri(self, model_name):
            return f"models:/{model_name}@champion"

        def list_versions(self, model_name):
            return [{"model_name": model_name, "version": "1"}]

    registry = ConcreteRegistry()

    assert (
        registry.register_model(
            "runs:/abc/model",
            "test-model",
            "abc",
            True,
        )["version"]
        == "1"
    )
