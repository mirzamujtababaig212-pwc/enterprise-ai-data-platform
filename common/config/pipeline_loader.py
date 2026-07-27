import yaml


class PipelineLoader:

    @staticmethod
    def load(name):

        with open(f"config/pipelines/{name}.yaml") as f:
            return yaml.safe_load(f)
