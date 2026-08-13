from common.factories.pipeline_factory import PipelineFactory


def test_silver_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline(
        "silver",
        spark,
    )
    df = spark.createDataFrame(
        [
            (
                "V1",
                "running",
                "2024-01-01",
                40.0,
                50.0,
                80.0,
                90.0,
            ),
            (
                "V2",
                "stopped",
                "2024-01-02",
                0.0,
                70.0,
                90.0,
                85.0,
            ),
        ],
        [
            "vehicle_id",
            "status",
            "event_time",
            "speed",
            "fuel_level",
            "battery",
            "engine_temperature",
        ],
    )
    result = pipeline.transformer.transform(df)
    valid, invalid = pipeline.validator.validate(result)
    assert valid.count() == 2
    assert invalid.count() == 0
