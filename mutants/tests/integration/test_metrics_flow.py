from unittest.mock import MagicMock

from common.factories.pipeline_factory import PipelineFactory


def test_metrics_called(spark):
    pipeline = PipelineFactory.get_pipeline("bronze", spark)
    pipeline.metrics.record_batch = MagicMock()
    df = spark.createDataFrame([(1, "Alice")], ["id", "name"])
    pipeline.metrics.record_batch("bronze", 1, df)
    pipeline.metrics.record_batch.assert_called_once()
