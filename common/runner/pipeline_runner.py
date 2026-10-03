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
        run_id=None,
    ):

        pipeline = PipelineFactory.get_pipeline(
            name,
            spark,
            glue_synchronizer=glue_synchronizer,
            run_id=run_id,
        )

        return pipeline.run(mode=mode)
