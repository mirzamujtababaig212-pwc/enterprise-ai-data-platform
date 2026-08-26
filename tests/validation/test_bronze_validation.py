from common.factories.pipeline_factory import PipelineFactory


def create_kafka_df(spark, rows):
    return spark.createDataFrame(
        rows,
        [
            "key",
            "value",
            "topic",
            "partition",
            "offset",
            "timestamp",
        ],
    )


def test_bronze_validation_accepts_valid_record(spark):
    df = create_kafka_df(
        spark,
        [
            (
                "key-1",
                '{"vehicle_id":"V001",'
                '"event_time":"2024-01-01T10:00:00",'
                '"speed":65.5,"rpm":2500}',
                "vehicle-events",
                0,
                1,
                "2024-01-01 10:00:00",
            ),
        ],
    )

    pipeline = PipelineFactory.get_pipeline("bronze", spark)

    transformed = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(transformed)

    assert valid.count() == 1
    assert invalid.count() == 0

    row = valid.collect()[0]

    assert row["vehicle_id"] == "V001"
    assert row["event_time"] is not None
    assert row["speed"] == 65.5
    assert row["rpm"] == 2500


def test_bronze_validation_rejects_null_event_time(spark):
    df = create_kafka_df(
        spark,
        [
            (
                "key-1",
                '{"vehicle_id":"V001",' '"speed":65.5,"rpm":2500}',
                "vehicle-events",
                0,
                1,
                "2024-01-01 10:00:00",
            ),
        ],
    )

    pipeline = PipelineFactory.get_pipeline("bronze", spark)

    transformed = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(transformed)

    assert valid.count() == 0
    assert invalid.count() == 1


def test_bronze_validation_rejects_null_vehicle_id(spark):
    df = create_kafka_df(
        spark,
        [
            (
                "key-1",
                '{"event_time":"2024-01-01T10:00:00",' '"speed":65.5,"rpm":2500}',
                "vehicle-events",
                0,
                1,
                "2024-01-01 10:00:00",
            ),
        ],
    )

    pipeline = PipelineFactory.get_pipeline("bronze", spark)

    transformed = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(transformed)

    assert valid.count() == 0
    assert invalid.count() == 1


def test_bronze_validation_rejects_duplicate_records(spark):
    payload = (
        '{"vehicle_id":"V001",' '"event_time":"2024-01-01T10:00:00",' '"speed":65.5,"rpm":2500}'
    )

    df = create_kafka_df(
        spark,
        [
            (
                "key-1",
                payload,
                "vehicle-events",
                0,
                1,
                "2024-01-01 10:00:00",
            ),
            (
                "key-2",
                payload,
                "vehicle-events",
                0,
                2,
                "2024-01-01 10:00:01",
            ),
        ],
    )

    pipeline = PipelineFactory.get_pipeline("bronze", spark)

    transformed = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(transformed)

    assert valid.count() == 1
    assert invalid.count() == 1


def test_bronze_validation_rejects_invalid_json(spark):
    df = create_kafka_df(
        spark,
        [
            (
                "key-1",
                '{"vehicle_id":"V001",}',
                "vehicle-events",
                0,
                1,
                "2024-01-01 10:00:00",
            ),
        ],
    )

    pipeline = PipelineFactory.get_pipeline("bronze", spark)

    transformed = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(transformed)

    assert valid.count() == 0
    assert invalid.count() == 1


def test_bronze_validation_accepts_multiple_valid_records(spark):
    df = create_kafka_df(
        spark,
        [
            (
                "key-1",
                '{"vehicle_id":"V001",'
                '"event_time":"2024-01-01T10:00:00",'
                '"speed":65.5,"rpm":2500}',
                "vehicle-events",
                0,
                1,
                "2024-01-01 10:00:00",
            ),
            (
                "key-2",
                '{"vehicle_id":"V002",'
                '"event_time":"2024-01-01T10:01:00",'
                '"speed":72.0,"rpm":2800}',
                "vehicle-events",
                0,
                2,
                "2024-01-01 10:01:00",
            ),
        ],
    )

    pipeline = PipelineFactory.get_pipeline("bronze", spark)

    transformed = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(transformed)

    assert valid.count() == 2
    assert invalid.count() == 0
