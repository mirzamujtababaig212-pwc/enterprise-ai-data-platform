from pyspark.sql.types import *
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from common.readers.parquet_reader import ParquetReader

schema = StructType(
    [
        StructField("id", IntegerType()),
        StructField("name", StringType())
    ]
)

def test_empty_parquet_reader(
    spark,
    temp_dir
):
    empty_df = spark.createDataFrame(
        [],
        schema
    )
    empty_df.write.mode("overwrite").parquet(temp_dir)
    reader = ParquetReader(
        path=temp_dir,
        schema=schema
    )
    df = reader.read(spark)
    assert df.count() == 0

def test_parquet_reader(
    spark,
    temp_dir
):
    df = spark.createDataFrame(
        [
            (1, "A"),
            (2, "B")
        ],
        schema
    )
    df.write.parquet(
        temp_dir,
        mode="overwrite"
    )
    reader = ParquetReader(
        path=temp_dir,
        schema=schema
    )
    result = reader.read(spark)
    assert result.count() == 2
    assert result.schema == schema

def test_large_parquet_reader(
    spark,
    temp_dir
):
    rows = [
        (i, f"name{i}")
        for i in range(1000)
    ]
    df = spark.createDataFrame(
        rows,
        schema
    )
    df.write.mode("overwrite").parquet(temp_dir)
    reader = ParquetReader(
        path=temp_dir,
        schema=schema
    )
    result = reader.read(spark)
    assert result.count() == 1000
