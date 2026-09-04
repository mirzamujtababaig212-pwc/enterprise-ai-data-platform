from __future__ import annotations

import sys

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression


TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "enterprise-ai-platform"


def main() -> int:
    print("=" * 70)
    print("MLFLOW MODEL LIFECYCLE TEST")
    print("=" * 70)

    mlflow.set_tracking_uri(TRACKING_URI)

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("MLflow version:", mlflow.__version__)

    mlflow.set_experiment(EXPERIMENT_NAME)

    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    accuracy = float(model.score(X, y))

    print("\nTraining model...")
    print("Model:", type(model).__name__)
    print("Accuracy:", accuracy)

    with mlflow.start_run(run_name="automated-model-lifecycle-test") as run:

        run_id = run.info.run_id

        mlflow.log_param(
            "model_type",
            "LogisticRegression",
        )

        mlflow.log_param(
            "test_environment",
            "docker",
        )

        mlflow.log_metric(
            "test_accuracy",
            accuracy,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
        )

        artifact_uri = mlflow.get_artifact_uri()

        print("\nRun created:")
        print("  Run ID:", run_id)
        print("  Experiment ID:", run.info.experiment_id)
        print("  Artifact URI:", artifact_uri)

    if not artifact_uri.startswith("s3://"):
        print("\nERROR: Model artifact URI is not S3.")
        return 1

    print("\nS3 artifact logging: PASSED")

    model_uri = f"runs:/{run_id}/model"

    print("\nLoading model:")
    print("  Model URI:", model_uri)

    loaded_model = mlflow.sklearn.load_model(model_uri)

    predictions = loaded_model.predict([[0], [1], [2], [3]])

    print("  Predictions:", predictions.tolist())

    expected = [0, 0, 1, 1]

    if predictions.tolist() != expected:
        print(f"\nERROR: Expected {expected}, " f"got {predictions.tolist()}")
        return 1

    print("\nModel download: PASSED")
    print("Model loading: PASSED")
    print("Model inference: PASSED")

    print("\n" + "=" * 70)
    print("MLFLOW MODEL LIFECYCLE TEST: PASSED")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
