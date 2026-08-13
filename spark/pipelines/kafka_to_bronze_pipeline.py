from common.config.settings import Settings
from common.pipelines.base_pipeline import BasePipeline
from common.pipelines.pipeline_config import PipelineConfig


class BronzePipeline(BasePipeline):

    CONFIG = PipelineConfig(
        pipeline_name="KafkaToBronze",
        source="kafka",
        target="delta",
        table=Settings.storage.BRONZE_TABLE,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
        query_name="kafka_to_bronze",
        output_mode="append",
        trigger={"processingTime": "10 seconds"},
        enable_validation=True,
        enable_metrics=True,
        enable_dlq=True,
        retries=3,
        retry_delay=2,
        execution_mode="stream",
    )
