from __future__ import annotations

from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.evaluator import ModelEvaluator
from ml.models.loan_default import (
    DEFAULT_MODEL_PARAMS,
    FEATURE_COLUMNS,
    MODEL_NAME,
    TARGET_COLUMN,
    LoanDefaultEvaluationAdapter,
    LoanDefaultMLP,
    validate_feature_columns,
)
from ml.platform import ModelMetadata, TrainingService
from ml.training.schemas import TrainingConfig, TrainingResult


class LoanDefaultTrainer(
    TrainingService[pd.DataFrame, TrainingResult],
):
    """Train and log the PyTorch loan-default classification model."""

    def train(
        self,
        dataframe: pd.DataFrame,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:

        if dataframe is None or dataframe.empty:
            raise ValueError("Loan default training dataframe must not be empty")

        validate_feature_columns(list(dataframe.columns))

        config = config or TrainingConfig()

        data = dataframe.copy()

        if TARGET_COLUMN not in data.columns:
            data[TARGET_COLUMN] = (
                (data["credit_score"] < 580)
                | (data["debt_to_income"] > 0.45)
                | ((data["loan_amount"] > data["income"] * 0.60) & (data["employment_years"] < 5))
            ).astype(int)

        required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]

        data = data[required_columns].dropna()

        if data.empty:
            raise ValueError("Loan default training data is empty after removing null values")

        if data[TARGET_COLUMN].nunique() < 2:
            raise ValueError("Loan default target must contain at least two classes")

        X = data[list(FEATURE_COLUMNS)].astype(np.float32)
        y = data[TARGET_COLUMN].astype(int)

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

        random_state = int(model_params["random_state"])

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_mean = X_train.mean(axis=0).to_numpy(dtype=np.float32)
        feature_std = X_train.std(axis=0).to_numpy(dtype=np.float32)

        feature_std = np.where(
            feature_std < 1e-8,
            1.0,
            feature_std,
        )

        model = LoanDefaultMLP(
            input_dim=len(FEATURE_COLUMNS),
            hidden_dim=int(model_params["hidden_dim"]),
            feature_mean=feature_mean,
            feature_std=feature_std,
        ).to(device)

        X_train_tensor = torch.tensor(
            X_train.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ).to(device)

        y_train_tensor = torch.tensor(
            y_train.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ).to(device)

        train_dataset = TensorDataset(
            X_train_tensor,
            y_train_tensor,
        )

        generator = torch.Generator()
        generator.manual_seed(random_state)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(model_params["batch_size"]),
            shuffle=True,
            generator=generator,
        )

        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(model_params["learning_rate"]),
        )

        epochs = int(model_params["epochs"])

        model.train()

        for _ in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                logits = model(batch_X)

                loss = criterion(
                    logits,
                    batch_y,
                )

                loss.backward()
                optimizer.step()

        model.eval()

        adapter = LoanDefaultEvaluationAdapter(model)

        evaluation = ModelEvaluator.evaluate(
            adapter,
            X_test,
            y_test,
        )

        metrics = evaluation.as_dict()

        model_params["device"] = str(device)

        mlflow.set_experiment(config.experiment_name)

        with mlflow.start_run(run_name=config.run_name) as run:

            mlflow.log_params(model_params)

            mlflow.log_params(
                {f"feature_mean_{index}": float(value) for index, value in enumerate(feature_mean)}
            )

            mlflow.log_params(
                {f"feature_std_{index}": float(value) for index, value in enumerate(feature_std)}
            )

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
                    "model_type": "MLP",
                    "framework": "pytorch",
                    "task_type": "binary_classification",
                    "target_column": TARGET_COLUMN,
                    "evaluation_type": "holdout",
                    "preprocessing": "standardization",
                }
            )

            model_info = mlflow.pytorch.log_model(
                model,
                name="model",
                serialization_format="pickle",
            )

            metadata = ModelMetadata(
                model_name=MODEL_NAME,
                model_type="MLP",
                framework="pytorch",
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
                    "model_type": "MLP",
                    "framework": "pytorch",
                    "task_type": "binary_classification",
                    "target_column": TARGET_COLUMN,
                    "evaluation_type": "holdout",
                    "preprocessing": "standardization",
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
