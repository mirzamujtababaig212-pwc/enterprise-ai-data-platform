from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.evaluation.evaluator import ModelEvaluator
from ml.models.customer_churn import (
    DEFAULT_MODEL_PARAMS,
    FEATURE_COLUMNS,
    MODEL_NAME,
    TARGET_COLUMN,
    validate_feature_columns,
)
from ml.platform import ModelMetadata, TrainingService
from ml.training.schemas import TrainingConfig, TrainingResult


class CustomerChurnTrainer(
    TrainingService[pd.DataFrame, TrainingResult],
):
    """Train and log the customer churn classification model."""

    def train(
        self,
        dataframe: pd.DataFrame,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        if dataframe is None or dataframe.empty:
            raise ValueError("Customer churn training dataframe must not be empty")

        validate_feature_columns(list(dataframe.columns))

        config = config or TrainingConfig()

        data = dataframe.copy()

        if TARGET_COLUMN not in data.columns:
            data[TARGET_COLUMN] = (
                (data["payment_failures"] >= 2)
                | (data["support_tickets"] >= 5)
                | ((data["tenure_months"] < 12) & (data["monthly_charges"] > 80))
            ).astype(int)

        required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
        data = data[required_columns].dropna()

        if data.empty:
            raise ValueError("Customer churn training data is empty after removing null values")

        if data[TARGET_COLUMN].nunique() < 2:
            raise ValueError("Customer churn target must contain at least two classes")

        X = data[list(FEATURE_COLUMNS)]
        y = data[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )

        model_params: dict[str, Any] = {
            **DEFAULT_MODEL_PARAMS,
            **config.model_params,
        }

        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)

        evaluation = ModelEvaluator.evaluate(
            model,
            X_test,
            y_test,
        )

        metrics = evaluation.as_dict()

        mlflow.set_experiment(config.experiment_name)

        with mlflow.start_run(run_name=config.run_name) as run:
            mlflow.log_params(model_params)

            mlflow.log_metrics(metrics)

            mlflow.log_metrics(
                {
                    "feature_count": float(len(FEATURE_COLUMNS)),
                    "training_samples": float(len(X_train)),
                    "validation_samples": float(len(X_test)),
                }
            )

            mlflow.set_tags(
                {
                    "model_name": MODEL_NAME,
                    "model_type": "LogisticRegression",
                    "framework": "scikit-learn",
                    "task_type": "binary_classification",
                    "target_column": TARGET_COLUMN,
                    "evaluation_type": "holdout",
                }
            )

            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
            )

            metadata = ModelMetadata(
                model_name=MODEL_NAME,
                model_type="LogisticRegression",
                framework="scikit-learn",
                task_type="binary_classification",
                target_column=TARGET_COLUMN,
                feature_names=FEATURE_COLUMNS,
                training_run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                model_uri=model_info.model_uri,
                metrics=metrics,
                parameters=model_params,
                tags={
                    "model_name": MODEL_NAME,
                    "model_type": "LogisticRegression",
                    "framework": "scikit-learn",
                    "task_type": "binary_classification",
                    "target_column": TARGET_COLUMN,
                    "evaluation_type": "holdout",
                },
            )

            return TrainingResult(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                model_uri=model_info.model_uri,
                metrics=metrics,
                parameters=model_params,
                training_samples=len(X_train),
                validation_samples=len(X_test),
                metadata=metadata,
            )
