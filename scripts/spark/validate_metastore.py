from __future__ import annotations

import sys
from pathlib import Path

from common.spark.spark_builder import SparkSessionBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE_DIR = PROJECT_ROOT / "spark-warehouse"
METASTORE_DIR = PROJECT_ROOT / "metastore_db"

DATABASE = "metastore_test"
TABLE = "persistence_check"
FULL_TABLE_NAME = f"{DATABASE}.{TABLE}"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    print_section("PROJECT PATHS")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Warehouse    : {WAREHOUSE_DIR}")
    print(f"Metastore    : {METASTORE_DIR}")

    if not WAREHOUSE_DIR.exists():
        print(f"ERROR: Warehouse directory does not exist: {WAREHOUSE_DIR}")
        return 1

    if not METASTORE_DIR.exists():
        print(f"ERROR: Metastore directory does not exist: {METASTORE_DIR}")
        return 1

    print_section("BUILD SPARK SESSION")

    spark = SparkSessionBuilder.build("MetastoreValidation")

    try:
        print(f"Spark version             : {spark.version}")
        print(
            "Catalog implementation    : "
            f"{spark.conf.get('spark.sql.catalogImplementation', 'NOT_SET')}"
        )
        print(
            "Warehouse directory       : " f"{spark.conf.get('spark.sql.warehouse.dir', 'NOT_SET')}"
        )
        print(
            "Delta catalog             : "
            f"{spark.conf.get('spark.sql.catalog.spark_catalog', 'NOT_SET')}"
        )

        print_section("DATABASES")

        databases = spark.sql("SHOW DATABASES")
        databases.show(truncate=False)

        database_names = {row[0] for row in databases.collect()}

        if DATABASE not in database_names:
            print(f"ERROR: Database '{DATABASE}' does not exist.")
            return 1

        print(f"PASS: Database '{DATABASE}' exists.")

        print_section("TABLES")

        tables = spark.sql(f"SHOW TABLES IN {DATABASE}")
        tables.show(truncate=False)

        table_names = {row["tableName"] for row in tables.collect()}

        if TABLE not in table_names:
            print(f"ERROR: Table '{FULL_TABLE_NAME}' does not exist.")
            return 1

        print(f"PASS: Table '{FULL_TABLE_NAME}' exists.")

        print_section("TABLE DATA")

        data = spark.sql(
            f"""
            SELECT id, name
            FROM {FULL_TABLE_NAME}
            ORDER BY id
            """
        )

        data.show(truncate=False)

        rows = [(row["id"], row["name"]) for row in data.collect()]

        expected_rows = [
            (1, "one"),
            (2, "two"),
            (3, "three"),
        ]

        if rows != expected_rows:
            print("ERROR: Persisted data does not match expected data.")
            print(f"Expected: {expected_rows}")
            print(f"Actual  : {rows}")
            return 1

        print("PASS: Persisted data matches expected data.")

        print_section("TABLE METADATA")

        detail = spark.sql(f"DESCRIBE DETAIL {FULL_TABLE_NAME}")

        detail.select(
            "format",
            "location",
        ).show(truncate=False)

        metadata = detail.collect()[0]

        table_format = metadata["format"]
        table_location = metadata["location"]

        if table_format.lower() != "delta":
            print("ERROR: Expected Delta table, " f"found '{table_format}'.")
            return 1

        print("PASS: Table format is Delta.")
        print(f"Table location: {table_location}")

        print_section("PHYSICAL STORAGE")

        delta_path = WAREHOUSE_DIR / f"{DATABASE}.db" / TABLE

        print(f"Expected Delta path: {delta_path}")

        if not delta_path.exists():
            print("ERROR: Expected Delta table directory " "does not exist.")
            return 1

        delta_log = delta_path / "_delta_log"

        if not delta_log.exists():
            print("ERROR: Delta transaction log directory " "does not exist.")
            return 1

        parquet_files = list(delta_path.glob("*.parquet"))
        delta_json_files = list(delta_log.glob("*.json"))

        print(f"Parquet files     : {len(parquet_files)}")
        print(f"Delta JSON files  : {len(delta_json_files)}")

        if not parquet_files:
            print("ERROR: No Parquet data files found.")
            return 1

        if not delta_json_files:
            print("ERROR: No Delta transaction log files found.")
            return 1

        print("PASS: Delta physical storage exists.")

        print_section("METASTORE FILES")

        metastore_files = list(METASTORE_DIR.rglob("*"))

        print(f"Metastore entries: " f"{len(metastore_files)}")

        if not metastore_files:
            print("ERROR: Metastore directory is empty.")
            return 1

        print("PASS: Metastore contains persisted state.")

        print_section("VALIDATION RESULT")

        print("ALL METASTORE VALIDATIONS PASSED.")
        print()
        print("Hive catalog persistence : PASS")
        print("Database persistence     : PASS")
        print("Table persistence        : PASS")
        print("Data persistence         : PASS")
        print("Delta storage            : PASS")
        print("Delta transaction log    : PASS")
        print("Metastore files          : PASS")

        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
