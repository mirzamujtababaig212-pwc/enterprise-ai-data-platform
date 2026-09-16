import os
import shutil
import tempfile

import boto3

import pytest
from delta import configure_spark_with_delta_pip
from dotenv import load_dotenv
from pyspark.sql import SparkSession


@pytest.fixture(scope="session", autouse=True)
def local_mlflow_environment():
    """
    Configure host-side integration tests for either local MLflow/MinIO
    or real AWS resources when AWS integration mode is enabled.
    """
    load_dotenv()

    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5051",
    )

    if os.getenv("RUN_AWS_INTEGRATION") == "1":
        os.environ.pop("MLFLOW_S3_ENDPOINT_URL", None)
    else:
        os.environ.setdefault(
            "MLFLOW_S3_ENDPOINT_URL",
            "http://localhost:9000",
        )

        os.environ.pop("AWS_PROFILE", None)

        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv(
            "MINIO_ROOT_USER",
            "minio",
        )

        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
            "MINIO_ROOT_PASSWORD",
            "minio123",
        )

        # boto3 may have initialized its default session during pytest
        # collection before the fixture configured the local MinIO credentials.
        boto3.DEFAULT_SESSION = None

    os.environ.setdefault(
        "AWS_DEFAULT_REGION",
        "us-east-1",
    )

    yield


@pytest.fixture(scope="session")
def spark():
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("Integration Tests")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "2")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def temp_dir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)
