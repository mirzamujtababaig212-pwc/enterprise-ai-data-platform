from __future__ import annotations

import os
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


class SparkSessionBuilder:
    """
    Central SparkSession factory for the Enterprise AI Platform.

    Supports:

    - Spark 4.0.1
    - Delta Lake 4.3.1
    - Spark Structured Streaming Kafka Consumer
    - Hive-compatible catalog
    - Local WSL2 execution
    - Project-local Spark warehouse

    IMPORTANT:
    Kafka and Delta dependencies are loaded centrally so that
    every batch and streaming pipeline receives the same Spark
    runtime configuration.
    """

    SPARK_VERSION = "4.0.1"
    SCALA_VERSION = "2.13"

    DELTA_PACKAGE = "io.delta:delta-spark_4.0_2.13:4.3.1"

    KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1"

    @staticmethod
    def build(
        app_name: str = "EnterpriseAIPlatform",
    ) -> SparkSession:

        # ==========================================================
        # PROJECT ROOT
        # ==========================================================

        project_root = Path(
            os.environ.get(
                "ENTERPRISE_AI_PLATFORM_ROOT",
                Path(__file__).resolve().parents[2],
            )
        ).resolve()

        # ==========================================================
        # SPARK WAREHOUSE
        # ==========================================================

        warehouse_dir = (project_root / "spark-warehouse").resolve()

        warehouse_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==========================================================
        # LOCAL / WSL2 NETWORKING
        # ==========================================================

        os.environ.setdefault(
            "SPARK_LOCAL_IP",
            "127.0.0.1",
        )

        # ==========================================================
        # SPARK BUILDER
        # ==========================================================

        builder = (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            # ------------------------------------------------------
            # Spark SQL warehouse
            # ------------------------------------------------------
            .config(
                "spark.sql.warehouse.dir",
                str(warehouse_dir),
            )
            # ------------------------------------------------------
            # Hive-compatible catalog
            # ------------------------------------------------------
            .config(
                "spark.sql.catalogImplementation",
                "hive",
            )
            # ------------------------------------------------------
            # Delta Lake
            # ------------------------------------------------------
            .config(
                "spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension",
            )
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            # ------------------------------------------------------
            # Performance / local development
            # ------------------------------------------------------
            .config(
                "spark.sql.shuffle.partitions",
                "4",
            )
            .config(
                "spark.default.parallelism",
                "4",
            )
            # ------------------------------------------------------
            # WSL2 driver networking
            # ------------------------------------------------------
            .config(
                "spark.driver.host",
                "127.0.0.1",
            )
            .config(
                "spark.driver.bindAddress",
                "127.0.0.1",
            )
            # ------------------------------------------------------
            # Local Spark UI
            # ------------------------------------------------------
            .config(
                "spark.ui.enabled",
                "false",
            )
        )

        # ==========================================================
        # DELTA DEPENDENCY
        # ==========================================================

        builder = configure_spark_with_delta_pip(builder)

        # -------------------------------------------------------------
        # Kafka Structured Streaming connector
        #
        # Spark 4.0.1
        # Scala 2.13
        # -------------------------------------------------------------

        existing_packages = builder._options.get("spark.jars.packages")

        kafka_package = "org.apache.spark:" "spark-sql-kafka-0-10_2.13:" "4.0.1"

        if existing_packages:
            packages = f"{existing_packages},{kafka_package}"
        else:
            packages = kafka_package

        builder = builder.config(
            "spark.jars.packages",
            packages,
        )

        # ==========================================================
        # CREATE SESSION
        # ==========================================================

        spark = builder.getOrCreate()

        # ==========================================================
        # LOGGING
        # ==========================================================

        spark.sparkContext.setLogLevel("WARN")

        return spark
