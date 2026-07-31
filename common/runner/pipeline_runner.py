from common.config.config_loader import ConfigLoader
from common.config.settings import Settings
from common.factories.pipeline_factory import PipelineFactory


class PipelineRunner:

    @staticmethod
    def run(name, spark):
        pipeline = PipelineFactory.get_pipeline(name, spark)
        pipeline.run()
