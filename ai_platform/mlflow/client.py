from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow import MlflowClient

from ai_platform.mlflow.config import MLflowConfig


logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Centralized MLflow integration for the Enterprise AI Platform.

    Responsibilities:
    - Configure MLflow tracking
    - Manage experiments
    - Create MLflow clients
    - Start runs
    - Expose model URIs
    """

    def __init__(self, config: MLflowConfig | None = None) -> None:
        self.config = config or MLflowConfig.from_environment()

        # Validate configuration before using it.
        self.config.validate()

        mlflow.set_tracking_uri(self.config.tracking_uri)

        self.client = MlflowClient(tracking_uri=self.config.tracking_uri)

        logger.info(
            "MLflow initialized",
            extra={
                "tracking_uri": self.config.tracking_uri,
                "experiment_name": self.config.experiment_name,
            },
        )

    @property
    def tracking_uri(self) -> str:
        return self.config.tracking_uri

    @property
    def experiment_name(self) -> str:
        return self.config.experiment_name

    def get_or_create_experiment(
        self,
        experiment_name: str | None = None,
    ) -> str:
        name = experiment_name or self.config.experiment_name

        experiment = self.client.get_experiment_by_name(name)

        if experiment is not None:
            return experiment.experiment_id

        return self.client.create_experiment(name)

    def set_experiment(
        self,
        experiment_name: str | None = None,
    ) -> str:
        name = experiment_name or self.config.experiment_name

        mlflow.set_experiment(name)

        experiment = self.client.get_experiment_by_name(name)

        if experiment is None:
            raise RuntimeError(f"MLflow experiment '{name}' could not be created.")

        return experiment.experiment_id

    def start_run(
        self,
        run_name: str | None = None,
        experiment_name: str | None = None,
        tags: dict[str, Any] | None = None,
    ):
        self.set_experiment(experiment_name)

        return mlflow.start_run(
            run_name=run_name,
            tags=tags,
        )

    def get_run(self, run_id: str):
        return self.client.get_run(run_id)

    def get_model_uri(
        self,
        run_id: str,
        model_name: str = "model",
    ) -> str:
        return f"runs:/{run_id}/{model_name}"

    def search_runs(
        self,
        experiment_name: str | None = None,
        max_results: int = 100,
    ):
        name = experiment_name or self.config.experiment_name

        experiment = self.client.get_experiment_by_name(name)

        if experiment is None:
            return []

        return self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=max_results,
        )
