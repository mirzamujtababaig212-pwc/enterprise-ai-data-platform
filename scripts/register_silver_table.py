from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DELTA_PATH = PROJECT_ROOT / "data" / "silver_delta"


def main():
    spark = SparkSessionBuilder.build("RegisterSilverTable")

    try:
        spark.sql("CREATE DATABASE IF NOT EXISTS silver")

        if spark.catalog.tableExists("silver.vehicle_events"):
            spark.sql("DROP TABLE silver.vehicle_events")

        spark.sql(
            f"""
            CREATE TABLE silver.vehicle_events
            USING DELTA
            LOCATION '{SILVER_DELTA_PATH}'
            """
        )

        print("Registered silver.vehicle_events")

        spark.sql("DESCRIBE EXTENDED silver.vehicle_events").show(
            100,
            truncate=False,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
