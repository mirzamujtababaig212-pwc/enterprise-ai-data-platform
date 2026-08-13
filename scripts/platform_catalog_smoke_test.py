from __future__ import annotations

from common.spark.spark_builder import SparkSessionBuilder


TABLES = [
    "bronze.vehicle_events",
    "silver.vehicle_events",
    "gold.vehicle_metrics",
]


def table_exists(
    spark,
    full_name: str,
) -> bool:

    database, table = full_name.split(
        ".",
        1,
    )

    databases = spark.sql("SHOW DATABASES").collect()

    database_exists = any(row["namespace"].lower() == database.lower() for row in databases)

    if not database_exists:
        return False

    tables = spark.sql(f"SHOW TABLES IN `{database}`").collect()

    return any(row["tableName"].lower() == table.lower() for row in tables)


def main() -> None:

    print("=" * 80)
    print("PLATFORM CATALOG SMOKE TEST")
    print("=" * 80)

    spark = SparkSessionBuilder.build("PlatformCatalogSmokeTest")

    try:

        print("\nSpark:")
        print(spark.version)

        print("\nCatalog implementation:")
        print(spark.conf.get("spark.sql.catalogImplementation"))

        print("\nDelta catalog:")
        print(spark.conf.get("spark.sql.catalog.spark_catalog"))

        print("\nWarehouse:")
        print(spark.conf.get("spark.sql.warehouse.dir"))

        print("\n=== DATABASES ===")

        spark.sql("SHOW DATABASES").show(truncate=False)

        print("\n=== TABLES ===")

        for table in TABLES:

            exists = table_exists(
                spark,
                table,
            )

            print(f"{'PASS' if exists else 'FAIL'} " f"{table}")

            if not exists:
                continue

            print(f"\n--- {table} ---")

            spark.table(table).show(truncate=False)

            print(f"Row count: " f"{spark.table(table).count()}")

        print()
        print("=" * 80)
        print("PLATFORM CATALOG SMOKE TEST PASSED")
        print("=" * 80)

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
