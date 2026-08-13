from __future__ import annotations

from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES = {
    "bronze.vehicle_events": (PROJECT_ROOT / "data" / "bronze" / "vehicle_events"),
    "silver.vehicle_events": (PROJECT_ROOT / "data" / "silver" / "vehicle_events"),
    "gold.vehicle_metrics": (PROJECT_ROOT / "data" / "gold" / "vehicle_metrics"),
}


def verify_table(spark, table_name: str, path: Path) -> None:
    print()
    print("=" * 80)
    print(f"VERIFYING: {table_name}")
    print("=" * 80)

    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(f"Delta path does not exist: {path}")

    df = spark.read.format("delta").load(str(path))

    print()
    print("Schema:")
    df.printSchema()

    count = df.count()

    print()
    print(f"Row count: {count}")

    print()
    print("Sample data:")
    df.show(
        20,
        truncate=False,
    )

    print()
    print("Delta history:")

    (
        spark.sql(f"DESCRIBE HISTORY delta.`{path}`")
        .select(
            "version",
            "timestamp",
            "operation",
            "operationMetrics",
        )
        .show(
            10,
            truncate=False,
        )
    )

    print()
    print(f"VERIFICATION PASSED: {table_name}")


def main() -> None:
    print("=" * 80)
    print("ENTERPRISE MEDALLION TABLE VERIFICATION")
    print("=" * 80)

    spark = SparkSessionBuilder.build("MedallionTableVerification")

    try:
        for table_name, path in TABLES.items():
            verify_table(
                spark,
                table_name,
                path,
            )

        print()
        print("=" * 80)
        print("ALL MEDALLION TABLES VERIFIED SUCCESSFULLY")
        print("=" * 80)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
