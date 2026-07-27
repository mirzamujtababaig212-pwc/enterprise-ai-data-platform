from common.pipelines.base_pipeline import BasePipeline


class DummyConfig:
    pipeline_name = "dummy"
    enable_validation = True
    enable_metrics = True
    enable_dlq = True
    retries = 2
    retry_delay = 0

class DummyPipeline(BasePipeline):
    CONFIG = DummyConfig()
