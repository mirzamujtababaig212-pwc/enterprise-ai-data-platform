import pytest

from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.parquet_reader import ParquetReader


def test_invalid_parquet_path(
    spark
):
    reader = ParquetReader(
        path="/does/not/exist"
    )
    with pytest.raises(Exception):
        reader.read(spark)

def test_invalid_csv_path(
    spark
):
    reader = CSVReader(
        path="/does/not/exist"
    )
    with pytest.raises(Exception):
        reader.read(spark)

def test_invalid_delta_path(
    spark
):
    reader = DeltaReader(
        path="/does/not/exist"
    )
    with pytest.raises(Exception):
        reader.read(spark)
