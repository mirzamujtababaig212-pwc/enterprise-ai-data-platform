from common.spark.spark_builder import SparkSessionBuilder

TABLE_NAME = "bronze.vehicle_events"


def main() -> None:
    spark = SparkSessionBuilder.build("ValidateBronze")

    try:
        print("=" * 80)
        print("BRONZE VALIDATION")
        print("=" * 80)

        df = spark.table(TABLE_NAME)

        print("\nSchema:")
        df.printSchema()

        print("\nData:")
        df.show(truncate=False)

        total_rows = df.count()

        distinct_vehicles = df.select("vehicle_id").distinct().count()

        null_vehicle_ids = df.filter(df.vehicle_id.isNull()).count()

        null_event_times = df.filter(df.event_time.isNull()).count()

        null_speeds = df.filter(df.speed.isNull()).count()

        print("\nValidation results:")
        print(f"Total rows       : {total_rows}")
        print(f"Distinct vehicles: {distinct_vehicles}")
        print(f"Null vehicle IDs : {null_vehicle_ids}")
        print(f"Null event times : {null_event_times}")
        print(f"Null speeds      : {null_speeds}")

        if total_rows != 12:
            raise AssertionError(f"Expected 12 Bronze rows, " f"found {total_rows}")

        if distinct_vehicles != 4:
            raise AssertionError(f"Expected 4 vehicles, " f"found {distinct_vehicles}")

        if null_vehicle_ids != 0:
            raise AssertionError("Bronze contains null vehicle IDs.")

        if null_event_times != 0:
            raise AssertionError("Bronze contains null event times.")

        if null_speeds != 0:
            raise AssertionError("Bronze contains null speeds.")

        print("\nBRONZE VALIDATION PASSED")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
