from __future__ import annotations

from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_DATABASE = "bronze"
BRONZE_TABLE = "vehicle_events"
BRONZE_FULL_TABLE = f"{BRONZE_DATABASE}.{BRONZE_TABLE}"

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "vehicle_events"


def main() -> None:
    print("=" * 80)
    print("REGISTER BRONZE DELTA TABLE")
    print("=" * 80)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Bronze path  : {BRONZE_PATH}")
    print(f"Exists       : {BRONZE_PATH.exists()}")
    print(f"Table        : {BRONZE_FULL_TABLE}")

    if not BRONZE_PATH.exists():
        raise FileNotFoundError(f"Bronze Delta path does not exist: {BRONZE_PATH}")

    spark = SparkSessionBuilder.build("RegisterBronzeTable")

    try:
        # ------------------------------------------------------------------
        # STEP 1: Ensure Bronze database exists
        # ------------------------------------------------------------------

        print("\nCreating Bronze database...")

        spark.sql(
            f"""
            CREATE DATABASE IF NOT EXISTS
            {BRONZE_DATABASE}
            """
        )

        print("Bronze database ready.")

        # ------------------------------------------------------------------
        # STEP 2: Register existing Delta path
        # ------------------------------------------------------------------

        print("\nRegistering Bronze Delta table...")

        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS
            {BRONZE_FULL_TABLE}
            USING DELTA
            LOCATION '{BRONZE_PATH}'
            """
        )

        print("Bronze table registration completed.")

        # ------------------------------------------------------------------
        # STEP 3: Verify table exists
        # ------------------------------------------------------------------

        print("\nBronze tables:")

        spark.sql(f"SHOW TABLES IN {BRONZE_DATABASE}").show(truncate=False)

        # ------------------------------------------------------------------
        # STEP 4: Read registered table
        # ------------------------------------------------------------------

        print("\nReading registered Bronze table:")

        bronze_df = spark.table(BRONZE_FULL_TABLE)

        bronze_df.printSchema()

        # ------------------------------------------------------------------
        # STEP 5: Count rows
        # ------------------------------------------------------------------

        bronze_count = bronze_df.count()

        print(f"\nBronze row count: {bronze_count}")

        # ------------------------------------------------------------------
        # STEP 6: Display data
        # ------------------------------------------------------------------

        print("\nBronze data:")

        bronze_df.show(truncate=False)

        # ------------------------------------------------------------------
        # STEP 7: Verify expected data
        # ------------------------------------------------------------------

        if bronze_count == 0:
            raise RuntimeError("Bronze table is registered but contains zero rows.")

        # ------------------------------------------------------------------
        # STEP 8: Verify Delta format
        # ------------------------------------------------------------------

        print("\nBronze table metadata:")

        (
            spark.sql(f"DESCRIBE DETAIL {BRONZE_FULL_TABLE}")
            .select(
                "format",
                "location",
                "numFiles",
                "sizeInBytes",
            )
            .show(truncate=False)
        )

        # ------------------------------------------------------------------
        # SUCCESS
        # ------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("BRONZE DELTA TABLE REGISTERED SUCCESSFULLY")
        print("=" * 80)

        print(f"Table      : {BRONZE_FULL_TABLE}")

        print(f"Location   : {BRONZE_PATH}")

        print(f"Row count  : {bronze_count}")

        print("=" * 80)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
