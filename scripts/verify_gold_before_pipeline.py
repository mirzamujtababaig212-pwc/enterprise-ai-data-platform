from __future__ import annotations

from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder


DATABASE = "gold"
TABLE = "vehicle_metrics"
FULL_TABLE = f"{DATABASE}.{TABLE}"

PROJECT_ROOT = Path.cwd().resolve()

EXPECTED_PATH = PROJECT_ROOT / "data" / "gold" / "vehicle_metrics"


def main() -> None:

    spark = SparkSessionBuilder.build("VerifyGoldBeforePipeline")

    try:

        print("=" * 80)
        print("GOLD TABLE PRE-PIPELINE VERIFICATION")
        print("=" * 80)

        print("\n=== EXPECTED PATH ===")

        print(EXPECTED_PATH)

        if not EXPECTED_PATH.exists():
            raise RuntimeError(f"Gold path does not exist: {EXPECTED_PATH}")

        if not (EXPECTED_PATH / "_delta_log").exists():
            raise RuntimeError(f"Gold path is not a Delta table: {EXPECTED_PATH}")

        print("\nPhysical Delta table: OK")

        print("\n=== CATALOG DETAIL ===")

        detail = spark.sql(f"DESCRIBE DETAIL {FULL_TABLE}")

        detail.select(
            "format",
            "location",
            "numFiles",
            "sizeInBytes",
        ).show(truncate=False)

        metadata = detail.collect()[0]

        actual_location = metadata["location"]

        expected_location = f"file:{EXPECTED_PATH}"

        if actual_location != expected_location:
            raise AssertionError(
                "Production Gold table is pointing to "
                "an unexpected location.\n"
                f"Expected: {expected_location}\n"
                f"Actual:   {actual_location}"
            )

        print("\nCatalog location: OK")

        print("\n=== GOLD DATA ===")

        gold_df = spark.table(FULL_TABLE)

        gold_df.show(truncate=False)

        row_count = gold_df.count()

        print(f"\nGold row count: {row_count}")

        print("\n=== GOLD SCHEMA ===")

        gold_df.printSchema()

        print("\n" + "=" * 80)
        print("GOLD PRE-PIPELINE VERIFICATION PASSED")
        print("=" * 80)

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
