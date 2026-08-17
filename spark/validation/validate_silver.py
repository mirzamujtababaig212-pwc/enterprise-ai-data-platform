from pathlib import Path

from pyspark.sql import functions as F

from common.spark.spark_builder import SparkSessionBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "kafka_vehicle_events"


REQUIRED_COLUMNS = [
    "vehicle_id",
    "event_time",
    "speed",
    "latitude",
    "longitude",
]


def fail(message: str) -> None:
    print()
    print(f"VALIDATION FAILED: {message}")
    raise AssertionError(message)


def main() -> None:

    spark = SparkSessionBuilder.build("SilverValidation")

    try:

        print("=" * 80)
        print("SILVER DATA QUALITY GATE")
        print("=" * 80)

        print(f"Silver path : {SILVER_PATH}")

        silver_df = spark.read.format("delta").load(str(SILVER_PATH))

        row_count = silver_df.count()

        print(f"Silver rows : {row_count}")

        if row_count == 0:
            fail("Silver dataset is empty.")

        # ------------------------------------------------------------------
        # Schema validation
        # ------------------------------------------------------------------

        print()
        print("SCHEMA VALIDATION")
        print("-" * 80)

        actual_columns = set(silver_df.columns)

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in actual_columns]

        if missing_columns:
            fail("Missing required columns: " + ", ".join(missing_columns))

        print("Required columns : PASS")

        # ------------------------------------------------------------------
        # Null vehicle_id
        # ------------------------------------------------------------------

        null_vehicle_ids = silver_df.filter(F.col("vehicle_id").isNull()).count()

        print(f"Null vehicle IDs : " f"{null_vehicle_ids}")

        if null_vehicle_ids > 0:
            fail("Silver contains null vehicle IDs.")

        # ------------------------------------------------------------------
        # Null event_time
        # ------------------------------------------------------------------

        null_event_times = silver_df.filter(F.col("event_time").isNull()).count()

        print(f"Null event times : " f"{null_event_times}")

        if null_event_times > 0:
            fail("Silver contains null event times.")

        # ------------------------------------------------------------------
        # Negative speed
        # ------------------------------------------------------------------

        negative_speed = silver_df.filter(F.col("speed") < 0).count()

        print(f"Negative speeds : " f"{negative_speed}")

        if negative_speed > 0:
            fail("Silver contains negative speeds.")

        # ------------------------------------------------------------------
        # Latitude
        # ------------------------------------------------------------------

        invalid_latitude = silver_df.filter(
            (F.col("latitude") < -90) | (F.col("latitude") > 90)
        ).count()

        print(f"Invalid latitude : " f"{invalid_latitude}")

        if invalid_latitude > 0:
            fail("Silver contains invalid latitude.")

        # ------------------------------------------------------------------
        # Longitude
        # ------------------------------------------------------------------

        invalid_longitude = silver_df.filter(
            (F.col("longitude") < -180) | (F.col("longitude") > 180)
        ).count()

        print(f"Invalid longitude : " f"{invalid_longitude}")

        if invalid_longitude > 0:
            fail("Silver contains invalid longitude.")

        # ------------------------------------------------------------------
        # Business duplicate validation
        # ------------------------------------------------------------------

        duplicate_groups = (
            silver_df.filter(F.col("vehicle_id").isNotNull() & F.col("event_time").isNotNull())
            .groupBy(
                "vehicle_id",
                "event_time",
            )
            .count()
            .filter(F.col("count") > 1)
        )

        duplicate_count = duplicate_groups.count()

        print(f"Duplicate business keys : " f"{duplicate_count}")

        if duplicate_count > 0:
            duplicate_groups.show(
                20,
                truncate=False,
            )

            fail("Duplicate business keys found.")

        # ------------------------------------------------------------------
        # Quality status
        # ------------------------------------------------------------------

        if "quality_status" in silver_df.columns:

            invalid_quality = silver_df.filter(
                ~F.col("quality_status").isin(
                    "VALID",
                    "SCHEMA_VARIATION_RECOVERED",
                )
            ).count()

            print(f"Invalid quality statuses : " f"{invalid_quality}")

            if invalid_quality > 0:
                fail("Invalid quality records found in Silver.")

        # ------------------------------------------------------------------
        # PASS
        # ------------------------------------------------------------------

        print()
        print("=" * 80)
        print("SILVER VALIDATION PASSED")
        print("=" * 80)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
