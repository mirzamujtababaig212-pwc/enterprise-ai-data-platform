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
                "2024-01-01",
                17.3850,
                78.4867,
                40.0,
                1500.0,
                50.0,
                80.0,
                90.0,
                3,
                "key-1",
                "vehicle-telemetry",
                0,
                100,
                "2024-01-01T00:00:00",
                '{"vehicle_id":"V1"}',
                "2024-01-01T00:00:01",
            ),
            (
                "V2",
                "2024-01-02",
                17.3851,
                78.4868,
                0.0,
                800.0,
                70.0,
                90.0,
                85.0,
                0,
                "key-2",
                "vehicle-telemetry",
                0,
                101,
                "2024-01-02T00:00:00",
                '{"vehicle_id":"V2"}',
                "2024-01-02T00:00:01",
            ),
        ],
        [
            "vehicle_id",
            "event_time",
            "latitude",
            "longitude",
            "speed",
            "rpm",
            "fuel_level",
            "battery",
            "engine_temperature",
            "gear",
            "kafka_key",
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
            "kafka_timestamp",
            "raw_value",
            "ingestion_time",
        ],
    )

    result = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(result)

    assert valid.count() == 2
    assert invalid.count() == 0
