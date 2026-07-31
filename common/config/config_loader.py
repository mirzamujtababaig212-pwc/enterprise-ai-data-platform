from pathlib import Path

import yaml


class ConfigLoader:

    @staticmethod
    def load(environment="dev"):

        config_path = Path("config") / "environments" / f"{environment}.yaml"

        with open(config_path) as f:
            return yaml.safe_load(f)
