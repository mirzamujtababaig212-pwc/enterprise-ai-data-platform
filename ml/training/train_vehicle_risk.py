from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import pandas as pd

from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


GOLD_PATH = os.getenv(
    "GOLD_PATH",
    "/app/data/lake/gold/vehicle_features",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)


FEATURE_COLUMNS = [
    "event_count",
    "avg_speed",
    "max_speed",
    "speed_stddev",
    "avg_rpm",
    "max_rpm",
    "avg_fuel_level",
    "min_fuel_level",
    "avg_battery",
    "avg_engine_temperature",
    "max_engine_temperature",
]


def create_spark() -> SparkSession:
    return SparkSession.builder.appName("vehicle-risk-training").getOrCreate()


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an initial deterministic risk label.

    This is a bootstrap label for the platform demonstration.
    A future version should replace this with a business-defined
    supervised target.
    """

    df = df.copy()

    df["risk"] = (
        (df["max_speed"] > 110) | (df["max_engine_temperature"] > 105) | (df["min_fuel_level"] < 15)
    ).astype(int)

    return df


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    spark = create_spark()

    spark_df = spark.read.format("delta").load(GOLD_PATH)

    pandas_df = spark_df.toPandas()

    pandas_df = create_labels(pandas_df)

    pandas_df = pandas_df.dropna(subset=FEATURE_COLUMNS + ["risk"])

    X = pandas_df[FEATURE_COLUMNS]
    y = pandas_df["risk"]

    if y.nunique() < 2:
        raise RuntimeError(
            "Training dataset contains only one class. " "Generate more telemetry before training."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "n_estimators",
            100,
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.sklearn.log_model(
            model,
            "vehicle-risk-model",
        )

        print(
            classification_report(
                y_test,
                predictions,
            )
        )

        print(f"Model accuracy: {accuracy:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()
