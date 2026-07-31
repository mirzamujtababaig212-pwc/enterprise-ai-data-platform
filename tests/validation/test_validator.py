from common.validation.validator import DataQualityValidator


def test_validator_returns_two_dataframes(spark):

    validator = DataQualityValidator()

    rows = [
        (
            "V1",
            "2025-01-01",
            60.0,
            80.0,
            90.0,
        )
    ]

    df = spark.createDataFrame(
        rows,
        [
            "vehicle_id",
            "timestamp",
            "speed",
            "fuel_level",
            "engine_temp",
        ],
    )

    valid_df, invalid_df = validator.validate(df)

    assert valid_df.count() == 1
    assert invalid_df.count() == 0


def test_validator_filters_invalid_speed(spark):

    validator = DataQualityValidator()

    rows = [
        (
            "V1",
            "2025-01-01",
            500.0,
            80.0,
            90.0,
        )
    ]

    df = spark.createDataFrame(
        rows,
        [
            "vehicle_id",
            "timestamp",
            "speed",
            "fuel_level",
            "engine_temp",
        ],
    )

    valid_df, invalid_df = validator.validate(df)

    assert valid_df.count() == 0
    assert invalid_df.count() == 1
