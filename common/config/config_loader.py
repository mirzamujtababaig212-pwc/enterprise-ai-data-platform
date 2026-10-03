from __future__ import annotations

import os
from pathlib import Path

import yaml


class ConfigLoader:
    @staticmethod
    def _resolve_root() -> Path:
        configured_root = os.getenv("ENTERPRISE_AI_PLATFORM_ROOT")

        if configured_root:
            return Path(configured_root)

        return Path(__file__).resolve().parents[2]

    @staticmethod
    def load(environment: str):
        root = ConfigLoader._resolve_root()
        config_path = root / "config" / "environments" / f"{environment}.yaml"

        with config_path.open(encoding="utf-8") as file:
            return yaml.safe_load(file)
