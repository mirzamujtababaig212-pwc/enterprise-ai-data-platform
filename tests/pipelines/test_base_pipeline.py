from unittest.mock import Mock

from common.pipelines.base_pipeline import BasePipeline
from tests.pipelines.dummy_pipeline import DummyPipeline


class DummyConfig:
    pipeline_name = "dummy"
    enable_validation = True
    enable_metrics = True
    enable_dlq = True
    retries = 1
    retry_delay = 0

class DummyPipeline(BasePipeline):
    CONFIG = DummyConfig()

reader = Mock()
validator = Mock()
writer = Mock()
metrics = Mock()
dlq = Mock()
transformer = Mock()
def test_pipeline_creation(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
        pipeline = DummyPipeline(
                spark=spark,
                reader=mock_reader,
                validator=mock_validator,
                writer=mock_writer,
                transformer=mock_transformer,
                metrics=mock_metrics,
                dlq=mock_dlq
        )

        assert pipeline.reader is mock_reader
        assert pipeline.writer is mock_writer
        assert pipeline.spark == spark

