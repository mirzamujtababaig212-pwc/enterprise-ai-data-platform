from __future__ import annotations

from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DATABASES = [
    "bronze",
    "silver",
    "gold",
    "control",
]

EXPECTED_TABLES = [
    "bronze.vehicle_events",
    "silver.vehicle_events",
    "gold.vehicle_metrics",
    "control.pipeline_run_history",
    "control.pipeline_stage_history",
]


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def database_exists(spark, database: str) -> bool:
    rows = spark.sql("SHOW DATABASES").collect()

    for row in rows:
        namespace = row["namespace"]

        if namespace.lower() == database.lower():
            return True

    return False


def table_exists(spark, full_table_name: str) -> bool:
    database, table = full_table_name.split(".", 1)

    if not database_exists(spark, database):
        return False

    rows = spark.sql(f"SHOW TABLES IN `{database}`").collect()

    for row in rows:
        if row["tableName"].lower() == table.lower():
            return True

    return False


def main() -> None:

    print_header("ENTERPRISE AI PLATFORM - METASTORE VERIFICATION")

    print(f"Project root: {PROJECT_ROOT}")

    spark = SparkSessionBuilder.build("EnterpriseMetastoreVerification")

    try:

        print_header("SPARK CONFIGURATION")

        print("Spark version:")
        print(spark.version)

        print("\nCatalog implementation:")
        print(spark.conf.get("spark.sql.catalogImplementation"))

        print("\nDelta catalog:")
        print(spark.conf.get("spark.sql.catalog.spark_catalog"))

        print("\nWarehouse:")
        print(spark.conf.get("spark.sql.warehouse.dir"))

        print_header("DATABASES")

        spark.sql("SHOW DATABASES").show(truncate=False)

        print_header("DATABASE VALIDATION")

        database_failures = []

        for database in EXPECTED_DATABASES:

            exists = database_exists(spark, database)

            status = "PASS" if exists else "FAIL"

            print(f"{status}: database={database}")

            if not exists:
                database_failures.append(database)

        print_header("TABLE VALIDATION")

        table_failures = []

        for table in EXPECTED_TABLES:

            exists = table_exists(spark, table)

            status = "PASS" if exists else "FAIL"

            print(f"{status}: table={table}")

            if not exists:
                table_failures.append(table)

        print_header("TABLE DETAILS")

        for table in EXPECTED_TABLES:

            if not table_exists(spark, table):
                continue

            print()
            print(f"TABLE: {table}")

            try:

                spark.sql(f"DESCRIBE DETAIL {table}").select("format", "location").show(
                    truncate=False
                )

            except Exception as exc:

                print(f"Could not read DESCRIBE DETAIL: {exc}")

        print_header("VALIDATION SUMMARY")

        if database_failures:
            print("Missing databases:")

            for database in database_failures:
                print(f"  - {database}")

        else:
            print("All expected databases: PASS")

        if table_failures:
            print("\nMissing tables:")

            for table in table_failures:
                print(f"  - {table}")

        else:
            print("All expected tables: PASS")

        if database_failures or table_failures:

            print()
            print("METASTORE VALIDATION: FAILED")

            raise SystemExit(1)

        print()
        print("METASTORE VALIDATION: PASSED")

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
