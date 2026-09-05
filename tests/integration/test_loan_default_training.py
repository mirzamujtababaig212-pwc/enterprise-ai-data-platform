from __future__ import annotations

import mlflow.pytorch
import pandas as pd
import pytest

from ml.training import LoanDefaultTrainer, TrainingConfig


def _loan_default_dataframe() -> pd.DataFrame:
    rows = []

    for index in range(60):
        rows.append(
            {
                "income": 30000.0 + (index % 10) * 5000.0,
                "age": 25 + (index % 35),
                "credit_score": 520 + (index % 20) * 15,
                "loan_amount": 10000.0 + (index % 12) * 5000.0,
                "employment_years": 1 + (index % 15),
                "debt_to_income": 0.20 + (index % 8) * 0.05,
                "default": int(index % 4 == 0 or index % 7 == 0),
            }
        )

    return pd.DataFrame(rows)


def test_loan_default_training_end_to_end() -> None:
    result = LoanDefaultTrainer().train(
        _loan_default_dataframe(),
        TrainingConfig(
            experiment_name="loan-default-model-tests",
            run_name="loan-default-training-test",
            model_params={
                "epochs": 5,
                "batch_size": 16,
                "random_state": 42,
            },
            test_size=0.3,
            random_state=42,
        ),
    )

    assert result.run_id
    assert result.experiment_id
    assert result.model_uri

    assert result.metadata is not None
    assert result.metadata.model_name == "LoanDefaultModel"
    assert result.metadata.model_type == "MLP"
    assert result.metadata.framework == "pytorch"
    assert result.metadata.task_type == "binary_classification"
    assert result.metadata.target_column == "default"
    assert result.metadata.feature_names == (
        "income",
        "age",
        "credit_score",
        "loan_amount",
        "employment_years",
        "debt_to_income",
    )

    assert result.metadata.training_run_id == result.run_id
    assert result.metadata.experiment_id == result.experiment_id
    assert result.metadata.model_uri == result.model_uri

    assert result.training_samples > 0
    assert result.validation_samples > 0

    assert "validation_accuracy" in result.metrics

    assert 0.0 <= result.metrics["validation_accuracy"] <= 1.0

    model = mlflow.pytorch.load_model(result.model_uri)

    assert model is not None
    assert type(model).__name__ == "LoanDefaultMLP"


def test_loan_default_training_rejects_missing_columns() -> None:
    trainer = LoanDefaultTrainer()

    with pytest.raises(ValueError, match="missing required"):
        trainer.train(
            pd.DataFrame(
                {
                    "income": [50000, 60000],
                    "age": [30, 40],
                }
            )
        )


def test_loan_default_training_rejects_single_class() -> None:
    dataframe = _loan_default_dataframe()

    dataframe["default"] = 0

    with pytest.raises(
        ValueError,
        match="at least two classes",
    ):
        LoanDefaultTrainer().train(dataframe)
