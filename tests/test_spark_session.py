from unittest.mock import MagicMock, patch

from common.spark_session import get_spark_session


@patch("common.spark_session.configure_spark_with_delta_pip")
@patch("common.spark_session.SparkSession")
def test_get_spark_session(mock_spark, mock_delta):
    builder = MagicMock()
    mock_spark.builder = builder
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.config.return_value = builder
    configured_builder = MagicMock()
    mock_delta.return_value = configured_builder
    spark = MagicMock()
    configured_builder.getOrCreate.return_value = spark
    result = get_spark_session("MyApp")
    builder.master.assert_called_once_with("local[2]")
    builder.appName.assert_called_once_with("Reader Tests")
    configured_builder.getOrCreate.assert_called_once()
    spark.sparkContext.setLogLevel.assert_called_once_with("WARN")
    assert result == spark
