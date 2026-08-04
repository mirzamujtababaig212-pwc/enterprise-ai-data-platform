import shutil
import tempfile

import pytest
from pyspark.sql import SparkSession


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
