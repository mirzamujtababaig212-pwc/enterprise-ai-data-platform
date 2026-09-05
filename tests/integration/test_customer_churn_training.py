import pandas as pd
import pytest

from ml.models.customer_churn import FEATURE_COLUMNS, MODEL_NAME, TARGET_COLUMN
from ml.training.customer_churn import CustomerChurnTrainer
from ml.training.schemas import TrainingConfig


def _customer_churn_dataframe() -> pd.DataFrame:
    rows = []

    for index in range(40):
        rows.append(
            {
                "tenure_months": 6 + (index % 30),
                "monthly_charges": 50.0 + (index % 8) * 8.0,
                "total_charges": 300.0 + index * 75.0,
                "support_tickets": index % 7,
                "usage_hours": 20.0 + (index % 10) * 5.0,
                "payment_failures": index % 4,
                "churn": int(index % 3 == 0),
            }
        )

    return pd.DataFrame(rows)


def test_customer_churn_training_returns_metadata() -> None:
    trainer = CustomerChurnTrainer()

    result = trainer.train(
        _customer_churn_dataframe(),
        TrainingConfig(
            experiment_name="customer-churn-test",
            run_name="customer-churn-training-test",
        ),
    )

    assert result.metadata is not None
    assert result.metadata.model_name == MODEL_NAME
    assert result.metadata.model_type == "LogisticRegression"
    assert result.metadata.framework == "scikit-learn"
    assert result.metadata.task_type == "binary_classification"
    assert result.metadata.target_column == TARGET_COLUMN
    assert result.metadata.feature_names == FEATURE_COLUMNS

    assert result.metadata.training_run_id == result.run_id
    assert result.metadata.experiment_id == result.experiment_id
    assert result.metadata.model_uri == result.model_uri


def test_customer_churn_training_supports_model_parameters() -> None:
    trainer = CustomerChurnTrainer()

    result = trainer.train(
        _customer_churn_dataframe(),
        TrainingConfig(
            experiment_name="customer-churn-test",
            run_name="customer-churn-params-test",
            model_params={
                "C": 0.5,
                "max_iter": 500,
            },
        ),
    )

    assert result.parameters["C"] == 0.5
    assert result.parameters["max_iter"] == 500


def test_customer_churn_training_rejects_single_class_target() -> None:
    dataframe = _customer_churn_dataframe()
    dataframe[TARGET_COLUMN] = 0

    with pytest.raises(ValueError, match="at least two classes"):
        CustomerChurnTrainer().train(dataframe)
