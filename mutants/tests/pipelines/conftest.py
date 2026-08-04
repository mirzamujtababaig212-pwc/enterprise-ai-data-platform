from unittest.mock import Mock

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.master("local[1]").appName("pipeline-tests").getOrCreate()
    yield spark
    spark.stop()


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
def mock_transformer():
    return Mock()


@pytest.fixture
def mock_metrics():
    return Mock()


@pytest.fixture
def mock_dlq():
    return Mock()
