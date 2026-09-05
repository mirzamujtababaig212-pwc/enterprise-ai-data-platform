from __future__ import annotations

import os
import time
from dataclasses import dataclass

import mlflow
import pandas as pd
from opentelemetry import trace

from ml.models.customer_churn import (
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
class CustomerChurnPrediction:
    """Immutable customer-churn prediction."""

    churn: int
    churn_probability: float | None
    model_name: str
    model_alias: str


class CustomerChurnPredictor(InferenceService[pd.DataFrame, CustomerChurnPrediction]):
    """
    Production inference wrapper for the Customer Churn model.

    The model is resolved from MLflow using an alias such as 'champion'.

    Observability is intentionally model/domain agnostic at the platform
    metric level:
      - request count
      - inference duration
      - classification count by predicted class
      - error count
      - OpenTelemetry traces
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
        """Load the model referenced by the configured MLflow alias."""

        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(
        self,
        features: pd.DataFrame,
    ) -> CustomerChurnPrediction:
        """Execute single-record inference with full observability."""

        start_time = time.perf_counter()

        with tracer.start_as_current_span("customer_churn.predict") as span:
            span.set_attribute("ml.model.name", self.model_name)
            span.set_attribute("ml.model.alias", self.model_alias)

            try:
                self._validate_features(features)

                if self.model is None:
                    with tracer.start_as_current_span("customer_churn.model_load") as load_span:
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

                prediction = self.model.predict(selected_features)
                churn = int(prediction[0])

                probability: float | None = None

                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(selected_features)

                    if probabilities.shape[1] >= 2:
                        probability = float(probabilities[0][1])

                span.set_attribute(
                    "ml.prediction.class",
                    churn,
                )

                if probability is not None:
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
                    prediction_class=str(churn),
                ).inc()

                return CustomerChurnPrediction(
                    churn=churn,
                    churn_probability=probability,
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

        predictions = self.model.predict(selected_features)

        result = features.copy()
        result["churn"] = predictions.astype(int)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(selected_features)

            if probabilities.shape[1] >= 2:
                result["churn_probability"] = probabilities[:, 1]

        result["model_name"] = self.model_name
        result["model_alias"] = self.model_alias

        return result

    @staticmethod
    def _validate_features(features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise ValueError("features must be a pandas DataFrame")

        if features.empty:
            raise ValueError("features must not be empty")

        validate_feature_columns(list(features.columns))

        if features[list(FEATURE_COLUMNS)].isnull().any().any():
            raise ValueError("features must not contain null values")
