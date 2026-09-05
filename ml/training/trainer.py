from __future__ import annotations

from typing import Any, Sequence

import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ai_platform.mlflow.client import MLflowManager

from ml.platform import TrainingService

from .schemas import TrainingConfig, TrainingResult


class ModelTrainer(
    TrainingService[
        tuple[Sequence[Sequence[float]], Sequence[int], BaseEstimator],
        TrainingResult,
    ]
):
    """
    Production training service.

    Responsibilities:
    - Train a model.
    - Record parameters in MLflow.
    - Record metrics in MLflow.
    - Persist the trained model to MLflow artifact storage.
    - Return a stable TrainingResult.

    Infrastructure concerns such as the MLflow tracking URI and
    artifact-store configuration remain encapsulated by MLflowManager.
    """

    def __init__(
        self,
        mlflow_manager: MLflowManager | None = None,
    ) -> None:
        self.mlflow_manager = mlflow_manager or MLflowManager()

    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        model: BaseEstimator,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        config = config or TrainingConfig()

        self._validate_dataset(X, y)

        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )

        with self.mlflow_manager.start_run(
            run_name=config.run_name,
            experiment_name=config.experiment_name,
        ) as run:

            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            validation_predictions = model.predict(X_validation)

            training_accuracy = float(
                accuracy_score(
                    y_train,
                    train_predictions,
                )
            )

            validation_accuracy = float(
                accuracy_score(
                    y_validation,
                    validation_predictions,
                )
            )

            parameters = self._extract_model_parameters(model)

            mlflow.log_params(parameters)

            mlflow.log_param(
                "test_size",
                config.test_size,
            )

            mlflow.log_param(
                "random_state",
                config.random_state,
            )

            mlflow.log_metric(
                "training_accuracy",
                training_accuracy,
            )

            mlflow.log_metric(
                "validation_accuracy",
                validation_accuracy,
            )

            mlflow.log_metric(
                "training_samples",
                len(X_train),
            )

            mlflow.log_metric(
                "validation_samples",
                len(X_validation),
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
            metrics={
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
            },
            parameters=parameters,
            training_samples=len(X_train),
            validation_samples=len(X_validation),
        )

    @staticmethod
    def _extract_model_parameters(
        model: BaseEstimator,
    ) -> dict[str, Any]:
        """
        Extract sklearn model parameters for MLflow logging.
        """

        parameters = model.get_params(deep=False)

        return {key: value for key, value in parameters.items() if value is not None}

    @staticmethod
    def _validate_dataset(
        X: Sequence[Sequence[float]],
        y: Sequence[int],
    ) -> None:
        if not X:
            raise ValueError("Training features X must not be empty")

        if not y:
            raise ValueError("Training labels y must not be empty")

        if len(X) != len(y):
            raise ValueError(
                "Training features and labels must contain " "the same number of samples"
            )

        feature_count = len(X[0])

        if feature_count == 0:
            raise ValueError("Training samples must contain at least one feature")

        for row in X:
            if len(row) != feature_count:
                raise ValueError("All training samples must have the same " "number of features")

        unique_labels = set(y)

        if len(unique_labels) < 2:
            raise ValueError("Training labels must contain at least two classes")
