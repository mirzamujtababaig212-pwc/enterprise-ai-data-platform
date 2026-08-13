from common.spark.spark_builder import SparkSessionBuilder

TABLE_NAME = "bronze.vehicle_events"


def main() -> None:
    spark = SparkSessionBuilder.build("VerifyBronzeTable")
    try:
        print("=" * 80)
        print("CATALOG")
        print("=" * 80)
        print(spark.conf.get("spark.sql.catalogImplementation"))
        print()
        print("=" * 80)
        print("DATABASES")
        print("=" * 80)
        spark.sql("SHOW DATABASES").show(truncate=False)
        print()
        print("=" * 80)
        print("TABLE EXISTS")
        print("=" * 80)
        exists = spark.catalog.tableExists(TABLE_NAME)
        print(exists)
        if not exists:
            raise RuntimeError(f"Table does not exist: {TABLE_NAME}")
        print()
        print("=" * 80)
        print("TABLE DETAILS")
        print("=" * 80)
        spark.sql(f"DESCRIBE DETAIL {TABLE_NAME}").show(
            truncate=False,
            vertical=True,
        )
        print()
        print("=" * 80)
        print("TABLE DATA")
        print("=" * 80)
        spark.table(TABLE_NAME).show(truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
