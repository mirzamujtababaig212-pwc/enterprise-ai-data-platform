from __future__ import annotations

from common.spark.spark_builder import SparkSessionBuilder


GOLD_TABLE = "gold.vehicle_metrics"


def main() -> None:
    spark = SparkSessionBuilder.build("VehicleDataQuality")

    failures: list[str] = []

    try:
        print("=" * 80)
        print("VEHICLE DATA QUALITY CHECKS")
        print("=" * 80)

        df = spark.table(GOLD_TABLE)

        # ------------------------------------------------------------------
        # CHECK 1: Table exists
        # ------------------------------------------------------------------
        print("\n[CHECK 1] Gold table exists")

        if not spark.catalog.tableExists(GOLD_TABLE):
            failures.append(f"Gold table does not exist: {GOLD_TABLE}")
        else:
            print("PASS")

        # ------------------------------------------------------------------
        # CHECK 2: Row count
        # ------------------------------------------------------------------
        print("\n[CHECK 2] Gold table has rows")

        row_count = df.count()

        print(f"Rows={row_count}")

        if row_count == 0:
            failures.append("Gold table contains zero rows")
        else:
            print("PASS")

        # ------------------------------------------------------------------
        # CHECK 3: vehicle_id not null
        # ------------------------------------------------------------------
        print("\n[CHECK 3] vehicle_id not null")

        null_vehicle_ids = df.filter(df.vehicle_id.isNull()).count()

        print(f"Null vehicle_id rows={null_vehicle_ids}")

        if null_vehicle_ids > 0:
            failures.append("Gold contains null vehicle_id values")
        else:
            print("PASS")

        # ------------------------------------------------------------------
        # CHECK 4: event_count positive
        # ------------------------------------------------------------------
        print("\n[CHECK 4] event_count positive")

        invalid_event_counts = df.filter((df.event_count <= 0) | df.event_count.isNull()).count()

        print(f"Invalid event_count rows={invalid_event_counts}")

        if invalid_event_counts > 0:
            failures.append("Gold contains invalid event_count values")
        else:
            print("PASS")

        # ------------------------------------------------------------------
        # CHECK 5: avg_speed valid
        # ------------------------------------------------------------------
        print("\n[CHECK 5] avg_speed valid")

        invalid_avg_speed = df.filter(df.avg_speed.isNull() | (df.avg_speed < 0)).count()

        print(f"Invalid avg_speed rows={invalid_avg_speed}")

        if invalid_avg_speed > 0:
            failures.append("Gold contains invalid avg_speed values")
        else:
            print("PASS")

        # ------------------------------------------------------------------
        # CHECK 6: min_speed <= avg_speed
        # ------------------------------------------------------------------
        print("\n[CHECK 6] min_speed <= avg_speed")

        invalid_min_avg = df.filter(df.min_speed > df.avg_speed).count()

        print(f"Invalid min/avg rows={invalid_min_avg}")

        if invalid_min_avg > 0:
            failures.append("min_speed is greater than avg_speed")
        else:
            print("PASS")

        # ------------------------------------------------------------------
        # CHECK 7: avg_speed <= max_speed
        # ------------------------------------------------------------------
        print("\n[CHECK 7] avg_speed <= max_speed")

        invalid_avg_max = df.filter(df.avg_speed > df.max_speed).count()

        print(f"Invalid avg/max rows={invalid_avg_max}")

        if invalid_avg_max > 0:
            failures.append("avg_speed is greater than max_speed")
        else:
            print("PASS")

        # ------------------------------------------------------------------
        # CHECK 8: first_event_time <= last_event_time
        # ------------------------------------------------------------------
        print("\n[CHECK 8] first_event_time <= last_event_time")

        invalid_timestamps = df.filter(df.first_event_time > df.last_event_time).count()

        print(f"Invalid timestamp rows={invalid_timestamps}")

        if invalid_timestamps > 0:
            failures.append("first_event_time is greater than " "last_event_time")
        else:
            print("PASS")

        # ------------------------------------------------------------------
        # RESULT
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("DATA QUALITY RESULT")
        print("=" * 80)

        if failures:
            print("FAILED")

            for failure in failures:
                print(f"- {failure}")

            raise RuntimeError("Data quality checks failed")

        print("ALL DATA QUALITY CHECKS PASSED")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
