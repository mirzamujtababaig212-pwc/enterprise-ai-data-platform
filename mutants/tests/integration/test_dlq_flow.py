from unittest.mock import MagicMock

from pyspark.sql.types import LongType, StringType, StructField, StructType

from common.factories.pipeline_factory import PipelineFactory

schema = StructType(
    [
        StructField("id", LongType(), True),
        StructField("email", StringType(), True),
    ]
)


def test_dlq_called(spark):
    pipeline = PipelineFactory.get_pipeline("bronze", spark)
    pipeline.dlq.write = MagicMock()
    invalid = spark.createDataFrame(
        [(1, None)],
        schema,
    )
    pipeline.handle_invalid_records(invalid)
    pipeline.dlq.write.assert_called_once()
