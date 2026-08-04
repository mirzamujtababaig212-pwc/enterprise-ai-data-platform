from common.pipelines.base_pipeline import BasePipeline
from common.pipelines.pipeline_config import PipelineConfig


class DummyConfig:
    pipeline_name = "dummy"
    enable_validation = True
    enable_metrics = True
    enable_dlq = True
    retries = 2
    retry_delay = 0


class DummyPipeline(BasePipeline):
    CONFIG = PipelineConfig(
        pipeline_name="Dummy",
        source="dummy",
        retries=3,
        retry_delay=0,
        enable_validation=True,
        enable_metrics=True,
        enable_dlq=True,
    )

    def cleanup(self):
        pass
