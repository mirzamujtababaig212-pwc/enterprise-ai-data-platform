from common.factories.pipeline_factory import PipelineFactory


def test_bronze_transform_validation(spark):
    pipeline = PipelineFactory.get_pipeline(
        "bronze",
        spark,
    )

    df = spark.createDataFrame(
        [
            (
                "key-1",
                '{"vehicle_id":"1"}',
                "vehicle-events",
                0,
                1,
                "2024-01-01 10:00:00",
            ),
            (
                "key-2",
                '{"vehicle_id":"2"}',
                "vehicle-events",
                0,
                2,
                "2024-01-01 10:01:00",
            ),
        ],
        [
            "key",
            "value",
            "topic",
            "partition",
            "offset",
            "timestamp",
        ],
    )

    transformed = pipeline.transformer.transform(df)

    assert transformed.count() == 2

    valid_df, invalid_df = pipeline.validator.validate(transformed)

    assert valid_df.count() == 2
    assert invalid_df.count() == 0
