from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from common.readers.csv_reader import CSVReader

schema = StructType([StructField("id", IntegerType()), StructField("name", StringType())])


def test_empty_csv_reader(spark, temp_dir):
    empty_df = spark.createDataFrame([], schema)
    empty_df.write.mode("overwrite").csv(temp_dir)
    reader = CSVReader(path=temp_dir, schema=schema)
    df = reader.read(spark)
    assert df.count() == 0


def test_csv_reader(spark, temp_dir):
    input_df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], schema)
    input_df.write.csv(temp_dir, header=True, mode="overwrite")
    reader = CSVReader(path=temp_dir, schema=schema)
    output_df = reader.read(spark)
    assert output_df.count() == 2
    assert output_df.columns == ["id", "name"]
    assert output_df.collect()[0]["name"] == "Alice"


def test_large_csv_reader(spark, temp_dir):
    rows = [(i, f"name{i}") for i in range(1000)]
    df = spark.createDataFrame(rows, schema)
    (df.write.option("header", True).mode("overwrite").csv(temp_dir))
    reader = CSVReader(path=temp_dir, schema=schema)
    result = reader.read(spark)
    assert result.count() == 1000
