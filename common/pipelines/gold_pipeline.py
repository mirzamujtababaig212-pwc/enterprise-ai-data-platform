from common.pipelines.base_pipeline import BasePipeline
from common.pipelines.pipeline_config import PipelineConfig


class GoldPipeline(BasePipeline):

    CONFIG = PipelineConfig(
        pipeline_name="Gold",
        source="silver",
        target="postgres",
        table="vehicle_metrics",
        enable_validation=False,
        enable_metrics=True,
        enable_dlq=False,
        retries=3,
        retry_delay=2,
    )

    def cleanup(self):
        pass
