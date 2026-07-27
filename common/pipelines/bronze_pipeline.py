from common.config.settings import Settings
from common.pipelines.base_pipeline import BasePipeline
from common.pipelines.pipeline_config import PipelineConfig


class BronzePipeline(BasePipeline):

    CONFIG = PipelineConfig(
        pipeline_name="Bronze",
        source="kafka",
        target="delta",
        table=Settings.storage.BRONZE_TABLE,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
        enable_validation=True,
        enable_metrics=True,
        enable_dlq=True,
        retries=3,
        retry_delay=2
    )

