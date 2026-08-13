from common.spark.spark_builder import SparkSessionBuilder

TABLE_NAME = "silver.vehicle_events"


def main() -> None:
    spark = SparkSessionBuilder.build("VerifySilverTable")
    try:
        print("=" * 80)
        print("SILVER TABLE")
        print("=" * 80)
        exists = spark.catalog.tableExists(TABLE_NAME)
        print(f"Exists: {exists}")
        if not exists:
            raise RuntimeError(f"Missing table: {TABLE_NAME}")
        print()
        print("=" * 80)
        print("SCHEMA")
        print("=" * 80)
        spark.table(TABLE_NAME).printSchema()
        print()
        print("=" * 80)
        print("ROW COUNT")
        print("=" * 80)
        count = spark.table(TABLE_NAME).count()
        print(count)
        print()
        print("=" * 80)
        print("DATA")
        print("=" * 80)
        (
            spark.table(TABLE_NAME)
            .orderBy(
                "vehicle_id",
                "event_time",
            )
            .show(truncate=False)
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
