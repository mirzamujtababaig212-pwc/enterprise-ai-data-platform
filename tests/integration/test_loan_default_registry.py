from __future__ import annotations

import mlflow
import mlflow.pytorch
import pandas as pd

from ml.registry import ModelRegistryManager
from ml.training import LoanDefaultTrainer, TrainingConfig


def _loan_default_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "income": 30000.0 + (index % 10) * 5000.0,
                "age": 25 + (index % 35),
                "credit_score": 520 + (index % 20) * 15,
                "loan_amount": 10000.0 + (index % 12) * 5000.0,
                "employment_years": 1 + (index % 15),
                "debt_to_income": 0.20 + (index % 8) * 0.05,
                "default": int(index % 4 == 0 or index % 7 == 0),
            }
            for index in range(60)
        ]
    )


def test_loan_default_registration_and_promotion() -> None:
    training_result = LoanDefaultTrainer().train(
        _loan_default_dataframe(),
        TrainingConfig(
            experiment_name="loan-default-registry-tests",
            run_name="loan-default-registry-test",
            model_params={
                "epochs": 5,
                "batch_size": 16,
                "random_state": 42,
            },
            test_size=0.3,
            random_state=42,
        ),
    )

    assert training_result.metadata is not None

    registry = ModelRegistryManager()

    registered = registry.register_model(
        model_uri=training_result.model_uri,
        model_name="LoanDefaultModel",
        run_id=training_result.run_id,
        evaluation_passed=True,
        metadata=training_result.metadata,
    )

    assert registered.model_name == "LoanDefaultModel"
    assert registered.alias == "candidate"

    candidate = registry.get_candidate("LoanDefaultModel")

    assert str(candidate.version) == registered.version
    assert candidate.tags.get("validation_status") == "PASSED"
    assert candidate.tags.get("model_type") == "MLP"
    assert candidate.tags.get("framework") == "pytorch"
    assert candidate.tags.get("task_type") == "binary_classification"
    assert candidate.tags.get("target_column") == "default"

    registry.promote_to_champion(
        model_name="LoanDefaultModel",
        version=registered.version,
    )

    champion = registry.get_champion("LoanDefaultModel")

    assert str(champion.version) == registered.version
    assert champion.tags.get("deployment_status") == "CHAMPION"

    uri = registry.get_champion_uri("LoanDefaultModel")

    assert uri == "models:/LoanDefaultModel@champion"

    mlflow.set_tracking_uri("http://127.0.0.1:5051")

    model = mlflow.pytorch.load_model(uri)

    assert model is not None
    assert type(model).__name__ == "LoanDefaultMLP"
