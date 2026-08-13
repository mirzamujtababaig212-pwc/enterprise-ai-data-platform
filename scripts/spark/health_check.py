from __future__ import annotations

import sys

from common.spark.spark_builder import SparkSessionBuilder


def main() -> int:
    print("=" * 80)
    print("SPARK HEALTH CHECK")
    print("=" * 80)

    spark = SparkSessionBuilder.build("SparkHealthCheck")

    try:
        print()
        print(f"Spark version          : {spark.version}")
        print(
            "Catalog implementation : "
            f"{spark.conf.get('spark.sql.catalogImplementation', 'NOT_SET')}"
        )
        print("Warehouse              : " f"{spark.conf.get('spark.sql.warehouse.dir', 'NOT_SET')}")
        print(
            "Spark catalog          : "
            f"{spark.conf.get('spark.sql.catalog.spark_catalog', 'NOT_SET')}"
        )

        print()
        print("=== DATABASES ===")

        spark.sql("SHOW DATABASES").show(truncate=False)

        print()
        print("=== SIMPLE SQL TEST ===")

        result = spark.sql(
            """
            SELECT
                1 AS id,
                'Spark is working' AS message
            """
        )

        result.show(truncate=False)

        print()
        print("SPARK HEALTH CHECK PASSED")

        return 0

    except Exception as exc:
        print()
        print("SPARK HEALTH CHECK FAILED")
        print(exc)
        return 1

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
