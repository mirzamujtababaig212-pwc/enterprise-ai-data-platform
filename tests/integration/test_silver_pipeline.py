from common.factories.pipeline_factory import PipelineFactory


def test_silver_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline(
        "silver",
        spark,
    )
    df = spark.createDataFrame(
        [
             ("V1","running","2024-01-01"),
             ("V2","stopped","2024-01-02"),
        ],
        [
             "vehicle_id",
             "status",
             "event_timestamp"
        ]
    )
    result = pipeline.transformer.transform(df)
    valid, invalid = pipeline.validator.validate(
        result
    )
    assert valid.count() == 2
    assert invalid.count() == 0
