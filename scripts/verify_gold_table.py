from common.spark.spark_builder import SparkSessionBuilder

TABLE_NAME = "gold.vehicle_metrics"


def main() -> None:
    spark = SparkSessionBuilder.build("VerifyGoldTable")
    try:
        print("=" * 80)
        print("GOLD TABLE")
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
        print("DATA")
        print("=" * 80)
        (spark.table(TABLE_NAME).orderBy("vehicle_id").show(truncate=False))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
