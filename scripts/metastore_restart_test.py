from __future__ import annotations

from common.spark.spark_builder import SparkSessionBuilder


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


def database_exists(spark, database: str) -> bool:

    databases = spark.sql("SHOW DATABASES").collect()

    return any(row["namespace"].lower() == database.lower() for row in databases)


def table_exists(spark, full_table_name: str) -> bool:

    database, table = full_table_name.split(".", 1)

    if not database_exists(spark, database):
        return False

    tables = spark.sql(f"SHOW TABLES IN `{database}`").collect()

    return any(row["tableName"].lower() == table.lower() for row in tables)


def main() -> None:

    print("=" * 80)
    print("METASTORE RESTART PERSISTENCE TEST")
    print("=" * 80)

    spark = SparkSessionBuilder.build("MetastoreRestartPersistenceTest")

    try:

        print("\nSpark version:")
        print(spark.version)

        print("\nCatalog:")
        print(spark.conf.get("spark.sql.catalogImplementation"))

        print("\nWarehouse:")
        print(spark.conf.get("spark.sql.warehouse.dir"))

        print("\n=== DATABASE PERSISTENCE ===")

        for database in EXPECTED_DATABASES:

            exists = database_exists(spark, database)

            print(f"{'PASS' if exists else 'FAIL'} " f"database: {database}")

        print("\n=== TABLE PERSISTENCE ===")

        for table in EXPECTED_TABLES:

            exists = table_exists(spark, table)

            print(f"{'PASS' if exists else 'FAIL'} " f"table: {table}")

        print("\n=== GOLD DATA ===")

        if table_exists(spark, "gold.vehicle_metrics"):

            gold_df = spark.table("gold.vehicle_metrics")

            gold_df.show(truncate=False)

            print(f"Gold row count: " f"{gold_df.count()}")

        else:

            print("gold.vehicle_metrics " "does not exist.")

        print("\n=== CONTROL RUN HISTORY ===")

        if table_exists(spark, "control.pipeline_run_history"):

            spark.table("control.pipeline_run_history").show(truncate=False)

        else:

            print("control.pipeline_run_history " "does not exist.")

        print("\n=== CONTROL STAGE HISTORY ===")

        if table_exists(spark, "control.pipeline_stage_history"):

            spark.table("control.pipeline_stage_history").show(truncate=False)

        else:

            print("control.pipeline_stage_history " "does not exist.")

        print()
        print("=" * 80)
        print("RESTART PERSISTENCE TEST COMPLETED")
        print("=" * 80)

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
