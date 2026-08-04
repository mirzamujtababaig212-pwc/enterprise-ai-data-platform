import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.master("local[2]").appName("Transformer Tests").getOrCreate()
    yield spark
    spark.stop()
