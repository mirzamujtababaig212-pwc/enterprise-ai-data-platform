from common.factories.pipeline_factory import (
    PipelineFactory,
)


class PipelineRunner:

    @staticmethod
    def run(name, spark, mode="stream"):

        pipeline = PipelineFactory.get_pipeline(
            name,
            spark,
        )

        return pipeline.run(mode=mode)
