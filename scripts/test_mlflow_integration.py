from __future__ import annotations

import sys

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "enterprise-ai-platform"


def main() -> int:
    print("=" * 70)
    print("MLFLOW INTEGRATION TEST")
    print("=" * 70)

    print(f"Tracking URI: {TRACKING_URI}")

    mlflow.set_tracking_uri(TRACKING_URI)

    print(f"MLflow version: {mlflow.__version__}")
    print(f"Effective tracking URI: {mlflow.get_tracking_uri()}")

    client = MlflowClient()

    print("\nChecking MLflow server...")

    experiments = client.search_experiments()

    print("MLflow connection: OK")
    print(
        "Experiments:",
        [(experiment.experiment_id, experiment.name) for experiment in experiments],
    )

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' does not exist yet.")
        print("Creating it...")

        experiment_id = client.create_experiment(EXPERIMENT_NAME)

        experiment = client.get_experiment(experiment_id)

    print("\nExperiment validation:")
    print("  ID:", experiment.experiment_id)
    print("  Name:", experiment.name)
    print("  Artifact location:", experiment.artifact_location)

    if not experiment.artifact_location.startswith("s3://"):
        print("\nERROR: Experiment is not using S3/MinIO artifacts.")
        return 1

    print("\nS3/MinIO artifact configuration: OK")

    print("\nMLFLOW INTEGRATION TEST: PASSED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
