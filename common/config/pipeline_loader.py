from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from common.config.settings import Settings
from common.pipelines.pipeline_runtime_config import PipelineRuntimeConfig


class PipelineLoader:
    """
    Loads pipeline YAML configuration and resolves references to
    Settings.storage values.

    Example:

        path: SILVER_PATH

    becomes:

        path: /app/data/delta/silver/vehicle_events
    """

    @staticmethod
    def _resolve_config_root() -> Path:
        configured_root = os.getenv("ENTERPRISE_AI_PLATFORM_ROOT")

        if configured_root:
            return Path(configured_root) / "config" / "pipelines"

        return Path(__file__).resolve().parents[2] / "config" / "pipelines"

    @staticmethod
    def load(name: str) -> dict[str, Any]:
        if not name or not name.strip():
            raise ValueError("Pipeline name cannot be empty.")

        pipeline_name = name.strip()

        config_path = PipelineLoader._resolve_config_root() / f"{pipeline_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline configuration not found: {config_path}")

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(f"Pipeline configuration must be a mapping: {config_path}")

        return PipelineLoader._resolve(deepcopy(config))

    @staticmethod
    def load_runtime_config(name: str) -> PipelineRuntimeConfig:
        """
        Load the typed runtime configuration for a pipeline.

        Runtime configuration is derived from the canonical pipeline YAML.
        Fields not explicitly configured use PipelineRuntimeConfig defaults.
        """

        config = PipelineLoader.load(name)

        pipeline = config.get("pipeline", {})
        runtime = config.get("runtime", {})

        if not isinstance(pipeline, dict):
            raise ValueError("Pipeline configuration 'pipeline' must be a mapping.")

        if not isinstance(runtime, dict):
            raise ValueError("Pipeline configuration 'runtime' must be a mapping.")

        pipeline_name = pipeline.get("class")

        if not pipeline_name:
            raise ValueError("Pipeline configuration requires 'pipeline.class'.")

        return PipelineRuntimeConfig(
            pipeline_name=pipeline_name,
            checkpoint=runtime.get("checkpoint", ""),
            output_mode=runtime.get("output_mode", "append"),
            query_name=runtime.get("query_name"),
            trigger=runtime.get("trigger"),
            retries=runtime.get("retries", 3),
            retry_delay=runtime.get("retry_delay", 2),
            enable_validation=runtime.get(
                "enable_validation",
                True,
            ),
            enable_metrics=runtime.get(
                "enable_metrics",
                True,
            ),
            enable_dlq=runtime.get(
                "enable_dlq",
                True,
            ),
        )

    @staticmethod
    def _resolve(value: Any) -> Any:
        """
        Recursively resolve configuration references.
        """

        if isinstance(value, dict):
            return {key: PipelineLoader._resolve(item) for key, item in value.items()}

        if isinstance(value, list):
            return [PipelineLoader._resolve(item) for item in value]

        if isinstance(value, str):
            return PipelineLoader._resolve_string(value)

        return value

    @staticmethod
    def _resolve_string(value: str) -> Any:
        """
        Resolve a string against supported Settings namespaces.

        Currently supported:

            BRONZE_PATH
            SILVER_PATH
            GOLD_PATH
            BRONZE_TABLE
            SILVER_TABLE
            GOLD_TABLE
            BRONZE_CHECKPOINT
            SILVER_CHECKPOINT
            etc.
        """

        storage = Settings.storage

        if hasattr(storage, value):
            return getattr(
                storage,
                value,
            )

        return value
