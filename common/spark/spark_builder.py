from __future__ import annotations

import os
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


class SparkSessionBuilder:
    """
    Central SparkSession factory for the Enterprise AI Platform.

    Supports:

    - Spark 4.1.1
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

    SPARK_VERSION = "4.1.1"
    SCALA_VERSION = "2.13"

    DELTA_PACKAGE = "io.delta:delta-spark_4.1_2.13:4.3.1"

    HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.4.2"

    KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"

    @staticmethod
    def build(
        app_name: str = "EnterpriseAIPlatform",
        include_kafka: bool = False,
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

        environment = os.environ.get("APP_ENV", "DEV").strip().lower()

        warehouse_dir = project_root / "spark-warehouse" / environment
        metastore_dir = project_root / "metastore_db" / environment

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
            .config(
                "spark.hadoop.javax.jdo.option.ConnectionURL",
                f"jdbc:derby:;databaseName={metastore_dir};create=true",
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

        # ==========================================================
        # AWS S3A DEPENDENCY
        # ==========================================================

        # Delta configuration sets spark.jars.packages, so Hadoop AWS
        # must be appended after configure_spark_with_delta_pip().
        existing_packages = builder._options.get("spark.jars.packages")
        hadoop_aws_package = SparkSessionBuilder.HADOOP_AWS_PACKAGE

        if existing_packages:
            packages = f"{existing_packages},{hadoop_aws_package}"
        else:
            packages = hadoop_aws_package

        builder = builder.config(
            "spark.jars.packages",
            packages,
        )

        # Local development uses the shared AWS CLI profile. AWS-hosted
        # runtimes can fall back to their attached IAM role credentials.
        builder = builder.config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )

        builder = builder.config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider,"
            "software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider",
        )

        # ==========================================================
        # Optional Kafka Structured Streaming connector
        #
        # Spark 4.1.1
        # Scala 2.13
        #
        # Kafka is intentionally opt-in. Batch and Delta-based
        # streaming workloads do not require the Kafka connector.
        # ==========================================================

        if include_kafka:
            existing_packages = builder._options.get("spark.jars.packages")
            kafka_package = SparkSessionBuilder.KAFKA_PACKAGE

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
