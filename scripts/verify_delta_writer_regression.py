from __future__ import annotations

from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder

DATABASE = "gold"
TABLE = "vehicle_metrics_writer_test"
FULL_TABLE = f"{DATABASE}.{TABLE}"

PROJECT_ROOT = Path.cwd().resolve()

EXPECTED_PATH = PROJECT_ROOT / "data" / "gold" / "vehicle_metrics_writer_test"


def main() -> None:

    spark = SparkSessionBuilder.build("VerifyDeltaWriterRegression")

    try:

        print("=" * 80)
        print("VERIFY DELTA WRITER REGRESSION")
        print("=" * 80)

        print("\n=== TABLE DETAIL ===")

        detail = spark.sql(f"DESCRIBE DETAIL {FULL_TABLE}")

        detail.select(
            "format",
            "location",
            "numFiles",
            "sizeInBytes",
        ).show(truncate=False)

        print("\n=== TABLE SCHEMA ===")

        spark.sql(f"DESCRIBE TABLE {FULL_TABLE}").show(truncate=False)

        print("\n=== TABLE DATA ===")

        spark.table(FULL_TABLE).show(truncate=False)

        print("\n=== EXPECTED PHYSICAL PATH ===")

        print(EXPECTED_PATH)

        if not EXPECTED_PATH.exists():
            raise RuntimeError(f"Expected Delta path missing: {EXPECTED_PATH}")

        if not (EXPECTED_PATH / "_delta_log").exists():
            raise RuntimeError("Delta _delta_log directory missing.")

        print("\nVerification successful.")

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
