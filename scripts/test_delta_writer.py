from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder
from common.writers.delta_writer import DeltaWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_PATH = PROJECT_ROOT / "data" / "silver_delta"
CHECKPOINT_PATH = PROJECT_ROOT / "spark" / "checkpoints" / "silver_delta"


def main():
    spark = SparkSessionBuilder.build("TestDeltaWriter")

    try:
        df = spark.createDataFrame(
            [
                (
                    "TEST-001",
                    "2026-08-11T00:00:00",
                    10.5,
                ),
            ],
            [
                "vehicle_id",
                "event_time",
                "speed",
            ],
        )

        print("=" * 80)
        print("INPUT DATA")
        print("=" * 80)

        df.printSchema()
        df.show(truncate=False)

        writer = DeltaWriter(
            table="silver.vehicle_events",
            path=str(SILVER_PATH),
            checkpoint=str(CHECKPOINT_PATH),
            mode="append",
            output_mode="append",
        )

        writer.write_batch(df)

        print()
        print("=" * 80)
        print("DELTA DATA")
        print("=" * 80)

        result = spark.read.format("delta").load(str(SILVER_PATH))

        result.printSchema()
        result.show(truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
