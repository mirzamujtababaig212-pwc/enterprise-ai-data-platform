from __future__ import annotations

import os
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5051",
)
EXPERIMENT_NAME = "integration-test-automated"


def test_mlflow_tracking_connection():
    """Verify MLflow tracking works without requiring a live server."""
    import tempfile
    from pathlib import Path

    previous_tracking_uri = mlflow.get_tracking_uri()

    with tempfile.TemporaryDirectory(prefix="mlflow-connection-test-") as temp_dir:
        database = Path(temp_dir).resolve() / "mlflow.db"
        tracking_uri = f"sqlite:///{database.as_posix()}"

        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.MlflowClient(tracking_uri=tracking_uri)
            experiments = client.search_experiments()

            assert experiments is not None
        finally:
            mlflow.set_tracking_uri(previous_tracking_uri)


def test_mlflow_s3_model_lifecycle():
    """Exercise a live MLflow server and S3 artifacts when explicitly enabled."""
    import pytest

    if os.getenv("RUN_MLFLOW_INTEGRATION") != "1":
        pytest.skip("Set RUN_MLFLOW_INTEGRATION=1 to run the live MLflow/S3 lifecycle test")

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
