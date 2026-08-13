from common.spark.spark_builder import SparkSessionBuilder


def main():
    spark = SparkSessionBuilder.build("CheckSilverTable")

    try:
        print("=" * 80)
        print("TABLE EXISTS")
        print("=" * 80)

        print(spark.catalog.tableExists("silver.vehicle_events"))

        print()
        print("=" * 80)
        print("TABLE DESCRIPTION")
        print("=" * 80)

        spark.sql("DESCRIBE EXTENDED silver.vehicle_events").show(
            100,
            truncate=False,
        )

        print()
        print("=" * 80)
        print("TABLE COUNT")
        print("=" * 80)

        print(spark.table("silver.vehicle_events").count())

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
