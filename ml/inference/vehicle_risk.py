from __future__ import annotations
import os
import time
from dataclasses import dataclass

import mlflow
import pandas as pd
from opentelemetry import trace

from ml.models.vehicle_risk import (
    FEATURE_COLUMNS,
    MODEL_NAME,
)
from ml.observability.metrics import (
    ML_INFERENCE_DURATION_SECONDS,
    ML_INFERENCE_ERRORS_TOTAL,
    ML_INFERENCE_PREDICTIONS_TOTAL,
    ML_INFERENCE_REQUESTS_TOTAL,
)


tracer = trace.get_tracer(__name__)


@dataclass(frozen=True)
class VehicleRiskPrediction:
    """
    Immutable vehicle-risk prediction.
    """

    risk: int
    risk_probability: float | None
    model_name: str
    model_alias: str


class VehicleRiskPredictor:
    """
    Production inference wrapper.

    The predictor resolves a registered MLflow model through
    an alias such as 'champion'.

    In addition to inference, this class records:
      - Prometheus request counters
      - Prometheus inference duration
      - Prometheus prediction counters
      - Prometheus error counters
      - OpenTelemetry tracing spans
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
        """
        Load the model referenced by the configured MLflow alias.
        """

        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(
        self,
        features: pd.DataFrame,
    ) -> VehicleRiskPrediction:
        """
        Execute single-record inference with full observability.
        """

        start_time = time.perf_counter()

        with tracer.start_as_current_span("vehicle_risk.predict") as span:

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

                    with tracer.start_as_current_span("vehicle_risk.model_load") as load_span:

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

                risk = int(prediction[0])

                probability: float | None = None

                if hasattr(
                    self.model,
                    "predict_proba",
                ):

                    probabilities = self.model.predict_proba(selected_features)

                    if probabilities.shape[1] >= 2:

                        probability = float(probabilities[0][1])

                # -------------------------------------------------
                # OpenTelemetry prediction attributes
                # -------------------------------------------------

                span.set_attribute(
                    "ml.prediction.risk",
                    risk,
                )

                if probability is not None:

                    span.set_attribute(
                        "ml.prediction.probability",
                        probability,
                    )

                # -------------------------------------------------
                # Prometheus metrics
                # -------------------------------------------------

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

                ML_INFERENCE_PREDICTIONS_TOTAL.labels(
                    model_name=self.model_name,
                    model_alias=self.model_alias,
                    risk=str(risk),
                ).inc()

                return VehicleRiskPrediction(
                    risk=risk,
                    risk_probability=probability,
                    model_name=self.model_name,
                    model_alias=self.model_alias,
                )

            except Exception as exc:

                error_type = type(exc).__name__

                # ---------------------------------------------
                # Error metrics
                # ---------------------------------------------

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

                # ---------------------------------------------
                # OpenTelemetry exception recording
                # ---------------------------------------------

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
        """
        Execute batch inference.

        Batch inference preserves the existing behavior and output
        contract. Single-record observability is implemented through
        predict(), while this method remains backward compatible.
        """

        self._validate_features(features)

        if self.model is None:
            self.load()

        selected_features = features[list(FEATURE_COLUMNS)]

        predictions = self.model.predict(selected_features)

        result = features.copy()

        result["risk"] = predictions.astype(int)

        if hasattr(
            self.model,
            "predict_proba",
        ):

            probabilities = self.model.predict_proba(selected_features)

            if probabilities.shape[1] >= 2:

                result["risk_probability"] = probabilities[:, 1]

        result["model_name"] = self.model_name

        result["model_alias"] = self.model_alias

        return result

    @staticmethod
    def _validate_features(
        features: pd.DataFrame,
    ) -> None:
        """
        Validate inference input.
        """

        if not isinstance(
            features,
            pd.DataFrame,
        ):

            raise TypeError("Inference features must be a pandas DataFrame")

        if features.empty:

            raise ValueError("Inference features must not be empty")

        missing = [column for column in FEATURE_COLUMNS if column not in features.columns]

        if missing:

            raise ValueError("Missing required inference features: " + ", ".join(missing))

        selected = features[list(FEATURE_COLUMNS)]

        if selected.isnull().any().any():

            raise ValueError("Inference features must not contain null values")
