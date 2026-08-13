from __future__ import annotations

import sys

from common.spark.spark_builder import SparkSessionBuilder


DATABASE = "metastore_roundtrip"
TABLE = "roundtrip_check"
FULL_TABLE_NAME = f"{DATABASE}.{TABLE}"


EXPECTED_ROWS = [
    (101, "alpha"),
    (102, "beta"),
    (103, "gamma"),
]


def build_spark(app_name: str):
    return SparkSessionBuilder.build(app_name)


def create_and_write() -> None:
    print()
    print("=" * 80)
    print("PHASE 1 — CREATE AND WRITE")
    print("=" * 80)

    spark = build_spark("MetastoreRoundtripWrite")

    try:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")

        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
                id INT,
                name STRING
            )
            USING DELTA
            """
        )

        spark.sql(f"DELETE FROM {FULL_TABLE_NAME}")

        values = ", ".join(f"({row_id}, '{name}')" for row_id, name in EXPECTED_ROWS)

        spark.sql(
            f"""
            INSERT INTO {FULL_TABLE_NAME}
            VALUES {values}
            """
        )

        print("Written rows:")
        spark.sql(
            f"""
            SELECT *
            FROM {FULL_TABLE_NAME}
            ORDER BY id
            """
        ).show(truncate=False)

    finally:
        spark.stop()

    print("Spark session stopped.")


def reopen_and_read() -> None:
    print()
    print("=" * 80)
    print("PHASE 2 — REOPEN AND READ")
    print("=" * 80)

    spark = build_spark("MetastoreRoundtripRead")

    try:
        databases = spark.sql("SHOW DATABASES")
        databases.show(truncate=False)

        tables = spark.sql(f"SHOW TABLES IN {DATABASE}")
        tables.show(truncate=False)

        result = spark.sql(
            f"""
            SELECT id, name
            FROM {FULL_TABLE_NAME}
            ORDER BY id
            """
        )

        result.show(truncate=False)

        actual_rows = [(row["id"], row["name"]) for row in result.collect()]

        if actual_rows != EXPECTED_ROWS:
            raise RuntimeError(
                "Roundtrip validation failed.\n"
                f"Expected: {EXPECTED_ROWS}\n"
                f"Actual  : {actual_rows}"
            )

        print("PASS: Data survived SparkSession restart.")

    finally:
        spark.stop()

    print("Spark session stopped.")


def main() -> int:
    try:
        create_and_write()
        reopen_and_read()

        print()
        print("=" * 80)
        print("ROUNDTRIP TEST PASSED")
        print("=" * 80)

        return 0

    except Exception as exc:
        print()
        print("=" * 80)
        print("ROUNDTRIP TEST FAILED")
        print("=" * 80)
        print(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
