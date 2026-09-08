import os
import shutil
import tempfile

import pytest
from dotenv import load_dotenv
from pyspark.sql import SparkSession


@pytest.fixture(scope="session", autouse=True)
def local_mlflow_environment():
    """
    Configure host-side integration tests to use the local MLflow/MinIO
    artifact store instead of falling through to real AWS credentials.
    """
    load_dotenv()

    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5051",
    )
    os.environ.setdefault(
        "MLFLOW_S3_ENDPOINT_URL",
        "http://localhost:9000",
    )

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv(
            "MINIO_ROOT_USER",
            "minio",
        )

    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
            "MINIO_ROOT_PASSWORD",
            "minio123",
        )

    os.environ.setdefault(
        "AWS_DEFAULT_REGION",
        "us-east-1",
    )

    yield


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("Integration Tests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def temp_dir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)
