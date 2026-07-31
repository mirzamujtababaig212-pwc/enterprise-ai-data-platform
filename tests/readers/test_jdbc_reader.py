from unittest.mock import MagicMock

from common.readers.jdbc_reader import JDBCReader


def test_jdbc_reader():
    spark = MagicMock()
    spark.read.jdbc.return_value = "df"
    reader = JDBCReader(
        url="jdbc:test",
        table="vehicle",
        properties={"user": "abc"},
    )
    result = reader.read(spark)
    spark.read.jdbc.assert_called_once_with(
        url="jdbc:test",
        table="vehicle",
        properties={"user": "abc"},
    )
    assert result == "df"
