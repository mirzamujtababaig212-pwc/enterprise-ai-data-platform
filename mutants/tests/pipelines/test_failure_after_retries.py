from unittest.mock import Mock

import pytest

from common.pipelines.base_pipeline import BasePipeline
from common.pipelines.pipeline_config import PipelineConfig


def test_failure_after_retries(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )
    batch = Mock()
    valid = Mock()
    invalid = Mock()
    transformed = Mock()
    mock_validator.validate.return_value = (valid, invalid)
    mock_transformer.transform.return_value = transformed
    mock_writer.write.side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError):
        pipeline.process_batch(batch, 5)
    assert mock_writer.write.call_count == 3


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

    def __init__(
        self,
        spark,
        reader,
        validator,
        writer,
        transformer,
        metrics,
        dlq,
    ):
        super().__init__(
            spark,
            reader,
            validator,
            writer,
            transformer,
            metrics,
            dlq,
        )
