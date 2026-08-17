from __future__ import annotations

from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder
from common.writers.delta_writer import DeltaWriter

TEST_DATABASE = "gold"
TEST_TABLE = "vehicle_metrics_writer_test"
TEST_FULL_TABLE = f"{TEST_DATABASE}.{TEST_TABLE}"

PROJECT_ROOT = Path.cwd().resolve()

TEST_PATH = PROJECT_ROOT / "data" / "gold" / "vehicle_metrics_writer_test"


def main() -> None:

    spark = SparkSessionBuilder.build("DeltaWriterRegressionTest")

    try:

        print("=" * 80)
        print("DELTA WRITER REGRESSION TEST")
        print("=" * 80)

        print(f"\nProject root : {PROJECT_ROOT}")
        print(f"Test table  : {TEST_FULL_TABLE}")
        print(f"Test path   : {TEST_PATH}")

        # ------------------------------------------------------------------
        # 1. Clean up only the isolated test table.
        # ------------------------------------------------------------------

        print("\n=== CLEANING TEST TABLE ===")

        spark.sql(f"DROP TABLE IF EXISTS {TEST_FULL_TABLE}")

        # Remove the physical test directory if it exists.
        if TEST_PATH.exists():
            import shutil

            shutil.rmtree(TEST_PATH)

        # ------------------------------------------------------------------
        # 2. Create deterministic test data.
        # ------------------------------------------------------------------

        print("\n=== CREATING TEST DATA ===")

        data = [
            ("TEST_V001", 3, 45.5),
            ("TEST_V002", 5, 55.2),
            ("TEST_V003", 2, 32.8),
        ]

        df = spark.createDataFrame(
            data,
            [
                "vehicle_id",
                "event_count",
                "avg_speed",
            ],
        )

        df.show(truncate=False)

        # ------------------------------------------------------------------
        # 3. Write using DeltaWriter.
        # ------------------------------------------------------------------

        print("\n=== WRITING WITH DELTAWRITER ===")

        writer = DeltaWriter(
            table=TEST_FULL_TABLE,
            path=str(TEST_PATH),
            mode="overwrite",
        )

        writer.write(df)

        # ------------------------------------------------------------------
        # 4. Verify physical Delta path.
        # ------------------------------------------------------------------

        print("\n=== VERIFYING PHYSICAL PATH ===")

        if not TEST_PATH.exists():
            raise RuntimeError(f"Expected Delta path does not exist: {TEST_PATH}")

        delta_log = TEST_PATH / "_delta_log"

        if not delta_log.exists():
            raise RuntimeError(f"Expected _delta_log does not exist: {delta_log}")

        print(f"Physical Delta path exists: {TEST_PATH}")

        print(f"Delta log exists: {delta_log}")

        # ------------------------------------------------------------------
        # 5. Verify catalog metadata.
        # ------------------------------------------------------------------

        print("\n=== VERIFYING CATALOG ===")

        detail = spark.sql(f"DESCRIBE DETAIL {TEST_FULL_TABLE}")

        detail.select(
            "format",
            "location",
            "numFiles",
            "sizeInBytes",
        ).show(truncate=False)

        metadata = detail.collect()[0]

        actual_format = metadata["format"]
        actual_location = metadata["location"]

        expected_location = f"file:{TEST_PATH}"

        print(f"Expected location: {expected_location}")

        print(f"Actual location  : {actual_location}")

        if actual_format != "delta":
            raise AssertionError(f"Expected Delta format, got: {actual_format}")

        if actual_location != expected_location:
            raise AssertionError(
                "Delta catalog location mismatch.\n"
                f"Expected: {expected_location}\n"
                f"Actual:   {actual_location}"
            )

        # ------------------------------------------------------------------
        # 6. Make sure Spark did NOT redirect the table into warehouse.
        # ------------------------------------------------------------------

        print("\n=== VERIFYING NO WAREHOUSE REDIRECTION ===")

        warehouse_path = PROJECT_ROOT / "spark-warehouse" / f"{TEST_DATABASE}.db" / TEST_TABLE

        print(f"Unexpected warehouse path: {warehouse_path}")

        if warehouse_path.exists():
            raise AssertionError(
                "DeltaWriter created the table inside " f"spark-warehouse: {warehouse_path}"
            )

        print("No unexpected spark-warehouse table detected.")

        # ------------------------------------------------------------------
        # 7. Verify table data.
        # ------------------------------------------------------------------

        print("\n=== VERIFYING TABLE DATA ===")

        result_df = spark.table(TEST_FULL_TABLE)

        result_df.show(truncate=False)

        row_count = result_df.count()

        print(f"Rows returned: {row_count}")

        if row_count != 3:
            raise AssertionError(f"Expected 3 rows, got {row_count}")

        # ------------------------------------------------------------------
        # 8. Verify schema.
        # ------------------------------------------------------------------

        print("\n=== VERIFYING SCHEMA ===")

        result_df.printSchema()

        expected_columns = [
            "vehicle_id",
            "event_count",
            "avg_speed",
        ]

        actual_columns = result_df.columns

        if actual_columns != expected_columns:
            raise AssertionError(
                "Schema mismatch.\n" f"Expected: {expected_columns}\n" f"Actual:   {actual_columns}"
            )

        # ------------------------------------------------------------------
        # 9. Final success.
        # ------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("DELTA WRITER REGRESSION TEST PASSED")
        print("=" * 80)

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
