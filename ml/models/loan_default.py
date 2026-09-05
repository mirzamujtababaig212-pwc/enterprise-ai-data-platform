from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import torch
from torch import nn


MODEL_NAME: Final[str] = "LoanDefaultModel"

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "income",
    "age",
    "credit_score",
    "loan_amount",
    "employment_years",
    "debt_to_income",
)

TARGET_COLUMN: Final[str] = "default"

DEFAULT_MODEL_PARAMS: Final[dict[str, int | float]] = {
    "hidden_dim": 16,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "random_state": 42,
}


class LoanDefaultMLP(nn.Module):
    """Small MLP for binary loan-default classification."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        hidden_dim: int = 16,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        if feature_mean is None:
            feature_mean = np.zeros(input_dim, dtype=np.float32)

        if feature_std is None:
            feature_std = np.ones(input_dim, dtype=np.float32)

        feature_mean_array = np.asarray(
            feature_mean,
            dtype=np.float32,
        )

        feature_std_array = np.asarray(
            feature_std,
            dtype=np.float32,
        )

        if feature_mean_array.shape != (input_dim,):
            raise ValueError("feature_mean must contain one value per input feature")

        if feature_std_array.shape != (input_dim,):
            raise ValueError("feature_std must contain one value per input feature")

        if np.any(feature_std_array <= 0):
            raise ValueError("feature_std must contain only positive values")

        self.register_buffer(
            "feature_mean",
            torch.from_numpy(feature_mean_array),
        )

        self.register_buffer(
            "feature_std",
            torch.from_numpy(feature_std_array),
        )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feature_mean) / self.feature_std
        return self.network(x).squeeze(-1)


class LoanDefaultEvaluationAdapter:
    """
    Adapter that exposes a scikit-learn-like prediction interface
    to the platform's existing ModelEvaluator.
    """

    def __init__(self, model: LoanDefaultMLP) -> None:
        self.model = model
        self.model.eval()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = X[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)

        tensor = torch.from_numpy(values)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack(
            (
                1.0 - probabilities,
                probabilities,
            )
        )


def validate_feature_columns(
    columns: list[str] | tuple[str, ...],
) -> None:
    """Validate that all required loan-default features are present."""

    missing = [column for column in FEATURE_COLUMNS if column not in columns]

    if missing:
        raise ValueError(f"missing required loan-default columns: {missing}")
