from __future__ import annotations

import mlflow

from ai_platform.mlflow.client import MLflowManager

from .observation_window import ObservationWindow

MONITORING_EXPERIMENT_NAME = "enterprise-ai-platform-monitoring"


def observation_artifact_path(window: ObservationWindow) -> str:
    return (
        "monitoring/observations/"
        f"{window.model_name}/"
        f"{window.model_version}/"
        f"{window.window_id}/"
        "observation_window.json"
    )


def persist_observation_window(
    window: ObservationWindow,
    *,
    mlflow_manager: MLflowManager | None = None,
) -> str:
    manager = mlflow_manager or MLflowManager()

    tags: dict[str, str] = {
        "monitoring_type": "observation_window",
        "observation_window_id": window.window_id,
        "model_name": window.model_name,
        "model_version": window.model_version,
        "model_alias": window.model_alias,
        "training_run_id": window.training_run_id,
        "feature_contract_name": window.feature_contract_name,
        "feature_contract_version": window.feature_contract_version,
        "window_start": window.window_start.isoformat(),
        "window_end": window.window_end.isoformat(),
    }

    with manager.start_run(
        run_name=f"observation-window-{window.window_id}",
        experiment_name=MONITORING_EXPERIMENT_NAME,
        tags=tags,
    ) as run:
        mlflow.log_dict(
            window.as_dict(),
            observation_artifact_path(window),
        )

        return run.info.run_id
