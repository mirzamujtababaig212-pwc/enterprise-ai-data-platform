
from common.factories.pipeline_factory import PipelineFactory


def test_bronze_transform_validation(spark):
    pipeline = PipelineFactory.get_pipeline(
        "bronze",
        spark,
    )
    df = spark.createDataFrame(
        [
             ('{"vehicle_id":"1"}',),
             ('{"vehicle_id":"2"}',),
        ],
        ["value"],
    )
    transformed = pipeline.transformer.transform(df)
    assert transformed.count() == 2
    valid_df, invalid_df = pipeline.validator.validate(
        transformed
    )
    assert valid_df.count() == 2
    assert invalid_df.count() == 0
