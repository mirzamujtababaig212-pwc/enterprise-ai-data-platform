from unittest.mock import MagicMock

from common.factories.pipeline_factory import PipelineFactory


def test_gold_writer_called(spark):
    pipeline = PipelineFactory.get_pipeline(
        "gold",
        spark,
    )
    pipeline.writer.write_batch = MagicMock()
    df = spark.createDataFrame([("A", 100), ("B", 200)], ["customer", "sales"])
    pipeline.writer.write_batch(df)
    pipeline.writer.write_batch.assert_called_once()
