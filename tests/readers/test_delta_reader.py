from pyspark.sql.types import *
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from common.readers.delta_reader import DeltaReader

schema = StructType(
    [
        StructField("id", IntegerType()),
        StructField("name", StringType())
    ]
)

def test_empty_delta_reader(
    spark,
    temp_dir
):
    empty_df = spark.createDataFrame(
        [],
        schema
    )
    (
        empty_df.write
                .format("delta")
                .mode("overwrite")
                .save(temp_dir)
    )
    reader = DeltaReader(
        path=temp_dir
    )
    df = reader.read(spark)
    assert df.count() == 0

def test_delta_reader(
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
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(temp_dir)
    )
    reader = DeltaReader(
        path=temp_dir
    )
    result = reader.read(spark)
    assert result.count() == 2
    assert result.columns == [
        "id",
        "name"
    ]

def test_large_delta_reader(
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
    (
        df.write
          .format("delta")
          .mode("overwrite")
          .save(temp_dir)
    )
    reader = DeltaReader(
        path=temp_dir,
        schema=schema
    )
    result = reader.read(spark)
    assert result.count() == 1000
