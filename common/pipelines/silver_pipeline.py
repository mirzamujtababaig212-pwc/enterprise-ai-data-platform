from common.config.settings import Settings
from common.pipelines.base_pipeline import BasePipeline
from common.pipelines.pipeline_config import PipelineConfig


class SilverPipeline(BasePipeline):

    CONFIG = PipelineConfig(
        pipeline_name="Silver",
        source="silver",
        table=Settings.storage.SILVER_TABLE,
        checkpoint=Settings.storage.SILVER_CHECKPOINT,
        enable_validation=True,
        enable_metrics=True,
        enable_dlq=True,
        retries=3,
        retry_delay=2,
    )
