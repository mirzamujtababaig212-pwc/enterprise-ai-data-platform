from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import functions as F

from common.spark.spark_builder import SparkSessionBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "kafka_vehicle_events"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "kafka_vehicle_events"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "quarantine" / "kafka_vehicle_events"


def main() -> None:
    spark = SparkSessionBuilder.build("BronzeSilverMetrics")

    started_at = datetime.now(UTC)

    try:
        bronze = spark.read.format("parquet").load(str(BRONZE_PATH))

        silver = spark.read.format("delta").load(str(SILVER_PATH))

        quarantine = spark.read.format("delta").load(str(QUARANTINE_PATH))

        bronze_count = bronze.count()
        silver_count = silver.count()
        quarantine_count = quarantine.count()

        represented = silver_count + quarantine_count

        dedup_removed = bronze_count - represented

        print("=" * 80)
        print("BRONZE → SILVER PIPELINE METRICS")
        print("=" * 80)

        print(f"Pipeline run time : {started_at.isoformat()}")

        print()

        print(f"Bronze rows               : {bronze_count}")

        print(f"Silver rows               : {silver_count}")

        print(f"Quarantine rows           : {quarantine_count}")

        print(f"Business duplicates removed : " f"{max(dedup_removed, 0)}")

        print(f"Silver + quarantine       : " f"{represented}")

        # ------------------------------------------------------------------
        # Quality distribution
        # ------------------------------------------------------------------

        if "quality_status" in quarantine.columns:

            print()
            print("QUALITY REJECTIONS")
            print("-" * 80)

            (
                quarantine.groupBy("quality_status")
                .count()
                .orderBy(F.col("count").desc())
                .show(truncate=False)
            )

        # ------------------------------------------------------------------
        # Silver quality
        # ------------------------------------------------------------------

        print()
        print("SILVER QUALITY")
        print("-" * 80)

        null_vehicle = silver.filter(F.col("vehicle_id").isNull()).count()

        null_event_time = silver.filter(F.col("event_time").isNull()).count()

        negative_speed = silver.filter(F.col("speed") < 0).count()

        print(f"Null vehicle IDs         : {null_vehicle}")

        print(f"Null event times         : {null_event_time}")

        print(f"Negative speeds          : {negative_speed}")

        # ------------------------------------------------------------------
        # Completion
        # ------------------------------------------------------------------

        print()
        print("=" * 80)
        print("METRICS COLLECTION COMPLETED")
        print("=" * 80)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
