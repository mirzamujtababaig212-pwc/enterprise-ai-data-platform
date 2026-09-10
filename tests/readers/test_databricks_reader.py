from unittest.mock import MagicMock

from common.readers.databricks_reader import DatabricksReader


def test_databricks_reader_stores_table():
    reader = DatabricksReader(
        table="vehicle_platform.rpt_vehicle_summary",
    )

    assert reader.table == "vehicle_platform.rpt_vehicle_summary"


def test_databricks_reader_reads_table():
    spark = MagicMock()

    dataframe = MagicMock()
    spark.table.return_value = dataframe

    reader = DatabricksReader(
        table="vehicle_platform.rpt_vehicle_summary",
    )

    result = reader.read(spark)

    spark.table.assert_called_once_with(
        "vehicle_platform.rpt_vehicle_summary",
    )

    assert result == dataframe
