from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ai_platform.mlflow.client import MLflowManager
from ml.evaluation import ModelEvaluator
from ml.models.vehicle_risk import (
    DEFAULT_MODEL_PARAMS,
    FEATURE_COLUMNS,
    MODEL_NAME,
    TARGET_COLUMN,
    validate_feature_columns,
)

from .schemas import (
    TrainingConfig,
    TrainingResult,
)


class VehicleRiskTrainer:
    """
    Production vehicle-risk model trainer.

    Responsibilities:

    1. Validate the training dataframe.
    2. Create deterministic bootstrap labels when required.
    3. Split data into training and validation datasets.
    4. Train RandomForest.
    5. Evaluate the model.
    6. Log parameters and metrics to MLflow.
    7. Persist the model to MLflow artifact storage.
    8. Return a stable TrainingResult.
    """

    def __init__(
        self,
        mlflow_manager: MLflowManager | None = None,
    ) -> None:

        self.mlflow_manager = mlflow_manager or MLflowManager()

    def train(
        self,
        dataframe: pd.DataFrame,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:

        config = config or TrainingConfig()

        dataframe = dataframe.copy()

        self._validate_dataframe(dataframe)

        if TARGET_COLUMN not in dataframe.columns:
            dataframe = self._create_bootstrap_labels(dataframe)

        dataframe = dataframe.dropna(
            subset=[
                *FEATURE_COLUMNS,
                TARGET_COLUMN,
            ]
        )

        if dataframe.empty:
            raise ValueError("Training dataframe contains no usable rows")

        X = dataframe[list(FEATURE_COLUMNS)]
        y = dataframe[TARGET_COLUMN]

        if y.nunique() < 2:
            raise ValueError("Training dataset contains only one target class")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )

        model_params: dict[str, Any] = dict(DEFAULT_MODEL_PARAMS)

        model_params["random_state"] = config.random_state

        model_params.update(config.model_params)

        model = RandomForestClassifier(**model_params)

        with self.mlflow_manager.start_run(
            run_name=config.run_name,
            experiment_name=config.experiment_name,
        ) as run:

            model.fit(
                X_train,
                y_train,
            )

            training_predictions = model.predict(X_train)

            training_accuracy = float((training_predictions == y_train.to_numpy()).mean())

            evaluation = ModelEvaluator.evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test,
            )

            metrics = {
                "training_accuracy": training_accuracy,
                **evaluation.as_dict(),
            }

            mlflow.log_params(model_params)

            mlflow.log_param(
                "feature_count",
                len(FEATURE_COLUMNS),
            )

            mlflow.log_param(
                "training_samples",
                len(X_train),
            )

            mlflow.log_param(
                "validation_samples",
                len(X_test),
            )

            mlflow.log_metrics(metrics)

            mlflow.set_tag(
                "model_name",
                MODEL_NAME,
            )

            mlflow.set_tag(
                "model_type",
                "RandomForestClassifier",
            )

            mlflow.set_tag(
                "target_column",
                TARGET_COLUMN,
            )

            mlflow.set_tag(
                "evaluation_type",
                "holdout",
            )

            mlflow.sklearn.log_model(
                model,
                name="model",
            )

            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            model_uri = self.mlflow_manager.get_model_uri(
                run_id=run_id,
                model_name="model",
            )

        return TrainingResult(
            run_id=run_id,
            experiment_id=experiment_id,
            model_uri=model_uri,
            metrics=metrics,
            parameters=model_params,
            training_samples=len(X_train),
            validation_samples=len(X_test),
        )

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe.empty:
            raise ValueError("Training dataframe must not be empty")

        validate_feature_columns(list(dataframe.columns))

    @staticmethod
    def _create_bootstrap_labels(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        dataframe = dataframe.copy()

        dataframe[TARGET_COLUMN] = (
            (dataframe["max_speed"] > 110)
            | (dataframe["max_engine_temperature"] > 105)
            | (dataframe["min_fuel_level"] < 15)
        ).astype(int)

        return dataframe
