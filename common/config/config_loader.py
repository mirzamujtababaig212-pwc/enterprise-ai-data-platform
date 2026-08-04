from pathlib import Path

import yaml


class ConfigLoader:

    @staticmethod
    def load(environment: str):
        root = Path(__file__).resolve().parents[2]
        config_path = root / "config" / "environments" / f"{environment}.yaml"

        with open(config_path) as f:
            return yaml.safe_load(f)
