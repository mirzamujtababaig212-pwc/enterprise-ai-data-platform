from pathlib import Path

import yaml


class PipelineLoader:

    CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "pipelines"

    @classmethod
    def load(cls, pipeline):

        file = cls.CONFIG_PATH / f"{pipeline.lower()}.yaml"

        if not file.exists():
            raise FileNotFoundError(file)

        with open(file, "r") as f:
            return yaml.safe_load(f)
