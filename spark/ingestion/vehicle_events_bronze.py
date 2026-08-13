from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder


DATABASE_NAME = "bronze"
TABLE_NAME = "vehicle_events"

SOURCE_FILE = Path("data/vehicle_events.csv")


def main() -> None:
    spark = SparkSessionBuilder.build("VehicleEventsBronzeIngestion")

    try:
        print("=" * 80)
        print("BRONZE INGESTION")
        print("=" * 80)

        if not SOURCE_FILE.exists():
            raise FileNotFoundError(f"Source file does not exist: {SOURCE_FILE}")

        print(f"Source file: {SOURCE_FILE}")

        spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")

        bronze_df = (
            spark.read.option("header", "true").option("inferSchema", "true").csv(str(SOURCE_FILE))
        )

        print("\nSource schema:")
        bronze_df.printSchema()

        print("\nSource data:")
        bronze_df.show(truncate=False)

        row_count = bronze_df.count()

        print(f"\nSource row count: {row_count}")

        if row_count == 0:
            raise ValueError("Source CSV contains zero rows.")

        required_columns = {
            "vehicle_id",
            "event_time",
            "speed",
        }

        missing_columns = required_columns - set(bronze_df.columns)

        if missing_columns:
            raise ValueError(
                "Source CSV is missing required columns: " f"{sorted(missing_columns)}"
            )

        (
            bronze_df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{DATABASE_NAME}.{TABLE_NAME}")
        )

        print("\nBronze table written successfully.")

        print(f"Table: " f"{DATABASE_NAME}.{TABLE_NAME}")

        print("\nBronze table contents:")

        (spark.table(f"{DATABASE_NAME}.{TABLE_NAME}").show(truncate=False))

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
