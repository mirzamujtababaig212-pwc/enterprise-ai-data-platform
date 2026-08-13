from common.config.settings import Settings
from common.pipelines.base_pipeline import BasePipeline
from common.pipelines.pipeline_config import PipelineConfig


class BatchBronzePipeline(BasePipeline):

    CONFIG = PipelineConfig(
        pipeline_name="BatchToBronze",
        source="parquet",
        path=Settings.storage.BATCH_INPUT_PATH,
        target="delta",
        table=Settings.storage.BRONZE_TABLE,
        output_mode="append",
        retries=3,
        retry_delay=2,
        enable_validation=True,
        enable_metrics=False,
        enable_dlq=False,
        execution_mode="batch",
    )
