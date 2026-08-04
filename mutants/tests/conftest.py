from unittest.mock import Mock

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("EnterprisePipelineTests")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    yield spark
    spark.stop()


@pytest.fixture
def sample_df(spark):

    return spark.createDataFrame([(1, "CarA"), (2, "CarB")], ["id", "name"])


@pytest.fixture
def mock_reader():
    return Mock()


@pytest.fixture
def mock_writer():
    return Mock()


@pytest.fixture
def mock_validator():
    return Mock()


@pytest.fixture
def mock_metrics():
    return Mock()


@pytest.fixture
def mock_dlq():
    return Mock()


@pytest.fixture
def mock_transformer():
    return Mock()
