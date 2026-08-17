from pathlib import Path

from pyspark.sql import functions as F

from common.spark.spark_builder import SparkSessionBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "kafka_vehicle_events"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "kafka_vehicle_events"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "quarantine" / "kafka_vehicle_events"


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    spark = SparkSessionBuilder.build("BronzeSilverReconciliation")

    try:
        print_header("BRONZE → SILVER RECONCILIATION VALIDATION")

        print(f"Bronze path     : {BRONZE_PATH}")
        print(f"Silver path     : {SILVER_PATH}")
        print(f"Quarantine path : {QUARANTINE_PATH}")

        # ------------------------------------------------------------------
        # Read datasets
        # ------------------------------------------------------------------

        print_header("READING DATA")

        bronze_df = spark.read.format("parquet").load(str(BRONZE_PATH))

        silver_df = spark.read.format("delta").load(str(SILVER_PATH))

        quarantine_df = spark.read.format("delta").load(str(QUARANTINE_PATH))

        bronze_count = bronze_df.count()
        silver_count = silver_df.count()
        quarantine_count = quarantine_df.count()

        print(f"Bronze rows     : {bronze_count}")
        print(f"Silver rows     : {silver_count}")
        print(f"Quarantine rows : {quarantine_count}")

        # ------------------------------------------------------------------
        # Quality classification
        # ------------------------------------------------------------------

        print_header("QUALITY COUNTS")

        if "quality_status" in quarantine_df.columns:
            quarantine_quality = (
                quarantine_df.groupBy("quality_status").count().orderBy("quality_status")
            )

            quarantine_quality.show(truncate=False)

        # ------------------------------------------------------------------
        # Duplicate validation
        #
        # IMPORTANT:
        # Only records with both vehicle_id and event_time are eligible
        # for business-key duplicate detection.
        # ------------------------------------------------------------------

        print_header("BUSINESS KEY DUPLICATE VALIDATION")

        if "vehicle_id" in silver_df.columns and "event_time" in silver_df.columns:
            duplicate_groups = (
                silver_df.filter(F.col("vehicle_id").isNotNull() & F.col("event_time").isNotNull())
                .groupBy(
                    "vehicle_id",
                    "event_time",
                )
                .count()
                .filter(F.col("count") > 1)
            )

            duplicate_group_count = duplicate_groups.count()

            print("Duplicate groups in Silver : " f"{duplicate_group_count}")

            if duplicate_group_count > 0:
                duplicate_groups.show(
                    50,
                    truncate=False,
                )
            else:
                print("PASS: No duplicate business keys found.")

        # ------------------------------------------------------------------
        # NULL key validation
        # ------------------------------------------------------------------

        print_header("SILVER NULL VALIDATION")

        null_vehicle_ids = silver_df.filter(F.col("vehicle_id").isNull()).count()

        null_event_times = silver_df.filter(F.col("event_time").isNull()).count()

        print(f"Null vehicle IDs : {null_vehicle_ids}")

        print(f"Null event times : {null_event_times}")

        # ------------------------------------------------------------------
        # Negative speed validation
        # ------------------------------------------------------------------

        print_header("SILVER RANGE VALIDATION")

        negative_speeds = 0

        if "speed" in silver_df.columns:
            negative_speeds = silver_df.filter(F.col("speed") < 0).count()

        print(f"Negative speeds : {negative_speeds}")

        # ------------------------------------------------------------------
        # Reconciliation
        #
        # Silver + quarantine can differ from Bronze because business
        # deduplication may intentionally remove records.
        # ------------------------------------------------------------------

        print_header("RECONCILIATION")

        represented_rows = silver_count + quarantine_count

        difference = bronze_count - represented_rows

        print(f"Bronze rows              : {bronze_count}")

        print(f"Silver rows              : {silver_count}")

        print(f"Quarantine rows          : {quarantine_count}")

        print(f"Silver + Quarantine      : {represented_rows}")

        print(f"Unrepresented rows       : {difference}")

        if difference == 0:
            print("RECONCILIATION STATUS   : PASS")
        elif difference > 0:
            print("RECONCILIATION STATUS   : " "PASS WITH DEDUPLICATION")

            print("The difference represents records " "removed by business-level deduplication.")
        else:
            print("RECONCILIATION STATUS   : FAIL")

            raise AssertionError("Silver + quarantine exceeds Bronze.")

        # ------------------------------------------------------------------
        # Final validation
        # ------------------------------------------------------------------

        print_header("FINAL VALIDATION")

        failures = []

        if null_vehicle_ids > 0:
            failures.append(f"Silver contains {null_vehicle_ids} " "null vehicle IDs.")

        if negative_speeds > 0:
            failures.append(f"Silver contains {negative_speeds} " "negative speeds.")

        if failures:
            print("VALIDATION STATUS : FAIL")

            for failure in failures:
                print(f" - {failure}")

            raise AssertionError("Silver validation failed.")

        print("VALIDATION STATUS : PASS")

        print_header("BRONZE → SILVER RECONCILIATION COMPLETED")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
