from __future__ import annotations

import os
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)
EXPERIMENT_NAME = "integration-test-automated"


def test_mlflow_tracking_connection():
    mlflow.set_tracking_uri(TRACKING_URI)

    client = mlflow.MlflowClient()

    experiments = client.search_experiments()

    assert experiments is not None


def test_mlflow_s3_model_lifecycle():
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    with mlflow.start_run(run_name="pytest-mlflow-test") as run:

        mlflow.log_param(
            "model_type",
            "LogisticRegression",
        )

        accuracy = float(model.score(X, y))

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
        )

        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri()

    assert artifact_uri.startswith("s3://")

    model_uri = f"runs:/{run_id}/model"

    loaded_model = mlflow.sklearn.load_model(model_uri)

    predictions = loaded_model.predict(X)

    assert predictions.tolist() == [0, 0, 1, 1]
