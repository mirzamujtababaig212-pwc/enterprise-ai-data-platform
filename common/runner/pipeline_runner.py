from common.factories.pipeline_factory import PipelineFactory


class PipelineRunner:
    @staticmethod
    def run(name, spark):
        pipeline = PipelineFactory.get_pipeline(name, spark)
        pipeline.run()
