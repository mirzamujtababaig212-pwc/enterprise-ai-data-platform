from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from common.config.settings import Settings


class PipelineLoader:
    """
    Loads pipeline YAML configuration and resolves references to
    Settings.storage values.

    Example:

        path: SILVER_PATH

    becomes:

        path: /home/annie/enterprise_ai_platform/data/delta/silver/vehicle_events
    """

    CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config" / "pipelines"

    @staticmethod
    def load(name: str) -> dict[str, Any]:
        if not name or not name.strip():
            raise ValueError("Pipeline name cannot be empty.")

        pipeline_name = name.strip()

        config_path = PipelineLoader.CONFIG_ROOT / f"{pipeline_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline configuration not found: " f"{config_path}")

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(f"Pipeline configuration must be a mapping: " f"{config_path}")

        return PipelineLoader._resolve(deepcopy(config))

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
