from common.factories.pipeline_factory import (
    PipelineFactory,
)


class PipelineRunner:

    @staticmethod
    def run(
        name,
        spark,
        mode="stream",
        *,
        glue_synchronizer=None,
    ):

        pipeline = PipelineFactory.get_pipeline(
            name,
            spark,
            glue_synchronizer=glue_synchronizer,
        )

        return pipeline.run(mode=mode)
