from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DELTA_PATH = PROJECT_ROOT / "data" / "silver_delta"


def main():
    print("=" * 80)
    print("PROJECT ROOT")
    print("=" * 80)
    print(PROJECT_ROOT)

    print()
    print("=" * 80)
    print("SILVER DELTA PATH")
    print("=" * 80)
    print(SILVER_DELTA_PATH)
    print("exists:", SILVER_DELTA_PATH.exists())

    spark = SparkSessionBuilder.build("VerifySilverDelta")

    try:
        print()
        print("=" * 80)
        print("SPARK VERSION")
        print("=" * 80)
        print(spark.version)

        print()
        print("=" * 80)
        print("DELTA PROVIDER TEST")
        print("=" * 80)

        test_df = spark.range(1)

        test_df.write.format("delta").mode("overwrite").save(str(SILVER_DELTA_PATH))

        print("Delta write succeeded.")

        print()
        print("=" * 80)
        print("DELTA DIRECTORY")
        print("=" * 80)

        delta_log = SILVER_DELTA_PATH / "_delta_log"

        print("Delta path:", SILVER_DELTA_PATH)
        print("Delta path exists:", SILVER_DELTA_PATH.exists())
        print("_delta_log exists:", delta_log.exists())

        if delta_log.exists():
            files = sorted(delta_log.iterdir())

            print()
            print("Delta log files:")

            for file in files[:20]:
                print("  ", file.name)

        print()
        print("=" * 80)
        print("REGISTERING silver.vehicle_events")
        print("=" * 80)

        spark.sql("CREATE DATABASE IF NOT EXISTS silver")

        if spark.catalog.tableExists("silver.vehicle_events"):
            print("Existing silver.vehicle_events found.")

            print()
            print("Current table description:")
            spark.sql("DESCRIBE EXTENDED silver.vehicle_events").show(100, truncate=False)

            print()
            print("Dropping ONLY the incorrect catalog registration.")
            print("This does NOT delete data/silver_delta.")

            spark.sql("DROP TABLE silver.vehicle_events")

        spark.sql(
            f"""
            CREATE TABLE silver.vehicle_events
            USING DELTA
            LOCATION '{SILVER_DELTA_PATH}'
            """
        )

        print()
        print("Table registered successfully.")

        print()
        print("=" * 80)
        print("FINAL TABLE DESCRIPTION")
        print("=" * 80)

        spark.sql("DESCRIBE EXTENDED silver.vehicle_events").show(100, truncate=False)

        print()
        print("=" * 80)
        print("TABLE CONTENT")
        print("=" * 80)

        spark.table("silver.vehicle_events").show()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
