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


def get_bronze_pipeline(spark):
    return PipelineFactory.get_pipeline("bronze", spark)


def test_schema_validator_accepts_expected_bronze_schema(spark):
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
            )
        ],
    )

    pipeline = get_bronze_pipeline(spark)

    transformed = pipeline.transformer.transform(df)

    schema_validator = pipeline.validator.validators[0]

    valid, invalid = schema_validator.validate(transformed)

    assert valid.count() == 1
    assert invalid is None


def test_null_validator_rejects_missing_event_time(spark):
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
            )
        ],
    )

    pipeline = get_bronze_pipeline(spark)

    transformed = pipeline.transformer.transform(df)

    null_validator = pipeline.validator.validators[1]

    valid, invalid = null_validator.validate(transformed)

    assert valid.count() == 0
    assert invalid.count() == 1


def test_null_validator_rejects_missing_vehicle_id(spark):
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
            )
        ],
    )

    pipeline = get_bronze_pipeline(spark)

    transformed = pipeline.transformer.transform(df)

    null_validator = pipeline.validator.validators[1]

    valid, invalid = null_validator.validate(transformed)

    assert valid.count() == 0
    assert invalid.count() == 1


def test_composite_validator_accepts_valid_record(spark):
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
            )
        ],
    )

    pipeline = get_bronze_pipeline(spark)

    transformed = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(transformed)

    assert valid.count() == 1
    assert invalid.count() == 0


def test_composite_validator_rejects_invalid_record(spark):
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
            )
        ],
    )

    pipeline = get_bronze_pipeline(spark)

    transformed = pipeline.transformer.transform(df)

    valid, invalid = pipeline.validator.validate(transformed)

    assert valid.count() == 0
    assert invalid.count() == 1
