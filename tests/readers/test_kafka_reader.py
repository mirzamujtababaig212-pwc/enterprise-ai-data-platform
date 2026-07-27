from unittest.mock import Mock

from common.readers.kafka_reader import KafkaReader


def __init__(self, **options):
        self.options = options

def test_kafka_reader(
    spark,
    mocker
):
    reader = KafkaReader(
        {
            "subscribe": "orders"
        }
    )
    dataframe = Mock()
    mock_read_stream = Mock()
    mocker.patch(
          "pyspark.sql.SparkSession.readStream",
          new_callable=mocker.PropertyMock,
          return_value=mock_read_stream,
    )
    (
        mock_read_stream.format
        .return_value
        .options
        .return_value
        .load
        .return_value
    ) = dataframe
    result = reader.read(
        spark
    )
    assert result == dataframe
