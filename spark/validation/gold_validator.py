from __future__ import annotations


from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark.schemas.gold_schema import GOLD_REQUIRED_COLUMNS


class GoldValidator:
    """
    Production-style validator for the Silver -> Gold layer.

    Validation is dynamic.

    No fixed dataset sizes are expected.
    """

    @staticmethod
    def validate_schema(
        gold_df: DataFrame,
    ) -> None:

        actual_columns = gold_df.columns

        missing = [column for column in GOLD_REQUIRED_COLUMNS if column not in actual_columns]

        if missing:
            raise AssertionError("Gold schema validation failed. " f"Missing columns: {missing}")

        extra = [column for column in actual_columns if column not in GOLD_REQUIRED_COLUMNS]

        if extra:
            raise AssertionError("Gold schema validation failed. " f"Unexpected columns: {extra}")

    @staticmethod
    def validate_not_null(
        gold_df: DataFrame,
    ) -> None:

        checks = [
            "vehicle_id",
            "event_count",
            "avg_speed",
            "min_speed",
            "max_speed",
            "first_event_time",
            "last_event_time",
        ]

        for column in checks:
            invalid_count = gold_df.filter(F.col(column).isNull()).count()

            if invalid_count != 0:
                raise AssertionError(
                    f"Gold NULL validation failed for " f"{column}: {invalid_count} null rows."
                )

    @staticmethod
    def validate_numeric_ranges(
        gold_df: DataFrame,
    ) -> None:

        invalid_event_count = gold_df.filter(F.col("event_count") < 1).count()

        if invalid_event_count != 0:
            raise AssertionError(
                "Gold event_count validation failed: "
                f"{invalid_event_count} rows have event_count < 1."
            )

        invalid_avg = gold_df.filter(F.col("avg_speed") < 0).count()

        if invalid_avg != 0:
            raise AssertionError(
                "Gold avg_speed validation failed: " f"{invalid_avg} negative rows."
            )

        invalid_min = gold_df.filter(F.col("min_speed") < 0).count()

        if invalid_min != 0:
            raise AssertionError(
                "Gold min_speed validation failed: " f"{invalid_min} negative rows."
            )

        invalid_max = gold_df.filter(F.col("max_speed") < 0).count()

        if invalid_max != 0:
            raise AssertionError(
                "Gold max_speed validation failed: " f"{invalid_max} negative rows."
            )

    @staticmethod
    def validate_speed_order(
        gold_df: DataFrame,
    ) -> None:

        invalid = gold_df.filter(F.col("min_speed") > F.col("avg_speed")).count()

        if invalid != 0:
            raise AssertionError("Gold speed ordering validation failed: " "min_speed > avg_speed.")

        invalid = gold_df.filter(F.col("avg_speed") > F.col("max_speed")).count()

        if invalid != 0:
            raise AssertionError("Gold speed ordering validation failed: " "avg_speed > max_speed.")

    @staticmethod
    def validate_time_order(
        gold_df: DataFrame,
    ) -> None:

        invalid = gold_df.filter(F.col("first_event_time") > F.col("last_event_time")).count()

        if invalid != 0:
            raise AssertionError(
                "Gold timestamp validation failed: " "first_event_time > last_event_time."
            )

    @staticmethod
    def validate_unique_vehicle(
        gold_df: DataFrame,
    ) -> None:

        duplicate_groups = gold_df.groupBy("vehicle_id").count().filter(F.col("count") > 1).count()

        if duplicate_groups != 0:
            raise AssertionError(
                "Gold uniqueness validation failed: "
                f"{duplicate_groups} duplicate vehicle groups."
            )

    @staticmethod
    def validate(
        gold_df: DataFrame,
    ) -> None:

        GoldValidator.validate_schema(gold_df)

        GoldValidator.validate_not_null(gold_df)

        GoldValidator.validate_numeric_ranges(gold_df)

        GoldValidator.validate_speed_order(gold_df)

        GoldValidator.validate_time_order(gold_df)

        GoldValidator.validate_unique_vehicle(gold_df)
