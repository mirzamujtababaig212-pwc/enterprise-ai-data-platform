from __future__ import annotations

import os
import time
from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from opentelemetry import trace

from ml.models.loan_default import (
    FEATURE_COLUMNS,
    MODEL_NAME,
    validate_feature_columns,
)
from ml.observability.metrics import (
    ML_INFERENCE_CLASSIFICATIONS_TOTAL,
    ML_INFERENCE_DURATION_SECONDS,
    ML_INFERENCE_ERRORS_TOTAL,
    ML_INFERENCE_REQUESTS_TOTAL,
)
from ml.platform import InferenceService

tracer = trace.get_tracer(__name__)


@dataclass(frozen=True)
class LoanDefaultPrediction:
    """Immutable loan-default prediction."""

    default: int
    default_probability: float
    model_name: str
    model_alias: str


class LoanDefaultPredictor(
    InferenceService[pd.DataFrame, LoanDefaultPrediction],
):
    """
    Production inference wrapper for the PyTorch Loan Default model.

    The model is resolved from MLflow using an alias such as 'champion'.

    The persisted PyTorch model contains its own feature-standardization
    parameters, so inference receives the same raw feature representation
    used by the training API.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_alias: str = "champion",
        tracking_uri: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_alias = model_alias

        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5051",
        )

        mlflow.set_tracking_uri(self.tracking_uri)

        self.model_uri = f"models:/{self.model_name}@{self.model_alias}"

        self.model = None

    def load(self) -> None:
        """Load the PyTorch model referenced by the configured MLflow alias."""

        self.model = mlflow.pytorch.load_model(self.model_uri)
        self.model.eval()

    def predict(
        self,
        features: pd.DataFrame,
    ) -> LoanDefaultPrediction:
        """Execute single-record inference with full observability."""

        start_time = time.perf_counter()

        with tracer.start_as_current_span("loan_default.predict") as span:
            span.set_attribute(
                "ml.model.name",
                self.model_name,
            )
            span.set_attribute(
                "ml.model.alias",
                self.model_alias,
            )

            try:
                self._validate_features(features)

                if self.model is None:
                    with tracer.start_as_current_span("loan_default.model_load") as load_span:
                        load_span.set_attribute(
                            "ml.model.name",
                            self.model_name,
                        )
                        load_span.set_attribute(
                            "ml.model.alias",
                            self.model_alias,
                        )
                        self.load()

                selected_features = features[list(FEATURE_COLUMNS)]

                values = selected_features.to_numpy(
                    dtype="float32",
                )

                tensor = torch.from_numpy(values)

                with torch.no_grad():
                    logits = self.model(tensor)
                    probabilities = torch.sigmoid(logits)

                probability = float(probabilities[0].item())
                prediction = int(probability >= 0.5)

                span.set_attribute(
                    "ml.prediction.class",
                    prediction,
                )
                span.set_attribute(
                    "ml.prediction.probability",
                    probability,
                )

                duration = time.perf_counter() - start_time

                ML_INFERENCE_DURATION_SECONDS.labels(
                    model_name=self.model_name,
                    model_alias=self.model_alias,
                ).observe(duration)

                ML_INFERENCE_REQUESTS_TOTAL.labels(
                    model_name=self.model_name,
                    model_alias=self.model_alias,
                    status="success",
                ).inc()

                ML_INFERENCE_CLASSIFICATIONS_TOTAL.labels(
                    model_name=self.model_name,
                    model_alias=self.model_alias,
                    prediction_class=str(prediction),
                ).inc()

                return LoanDefaultPrediction(
                    default=prediction,
                    default_probability=probability,
                    model_name=self.model_name,
                    model_alias=self.model_alias,
                )

            except Exception as exc:
                error_type = type(exc).__name__

                ML_INFERENCE_ERRORS_TOTAL.labels(
                    model_name=self.model_name,
                    model_alias=self.model_alias,
                    error_type=error_type,
                ).inc()

                ML_INFERENCE_REQUESTS_TOTAL.labels(
                    model_name=self.model_name,
                    model_alias=self.model_alias,
                    status="error",
                ).inc()

                span.record_exception(exc)
                span.set_attribute(
                    "ml.inference.error_type",
                    error_type,
                )

                raise

    def predict_batch(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Execute batch inference."""

        self._validate_features(features)

        if self.model is None:
            self.load()

        selected_features = features[list(FEATURE_COLUMNS)]

        values = selected_features.to_numpy(
            dtype="float32",
        )

        tensor = torch.from_numpy(values)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)

        probabilities_np = probabilities.cpu().numpy()

        predictions = (probabilities_np >= 0.5).astype(int)

        result = features.copy()

        result["default"] = predictions
        result["default_probability"] = probabilities_np

        result["model_name"] = self.model_name
        result["model_alias"] = self.model_alias

        return result

    @staticmethod
    def _validate_features(
        features: pd.DataFrame,
    ) -> None:
        if not isinstance(features, pd.DataFrame):
            raise ValueError("features must be a pandas DataFrame")

        if features.empty:
            raise ValueError("features must not be empty")

        validate_feature_columns(list(features.columns))

        if features[list(FEATURE_COLUMNS)].isnull().any().any():
            raise ValueError("features must not contain null values")
