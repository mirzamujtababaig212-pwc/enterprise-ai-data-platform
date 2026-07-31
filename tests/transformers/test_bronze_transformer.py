from pyspark.sql import Row
from pyspark.sql.types import StringType, StructField, StructType

from common.transformers.bronze_transformer import BronzeTransformer

schema = StructType([StructField("value", StringType(), True)])


class TestBronzeTransformer:

    def test_create(self):
        transformer = BronzeTransformer()
        assert transformer is not None

    def test_transform_returns_dataframe(self, spark):
        transformer = BronzeTransformer()
        df = spark.createDataFrame(
            [
                ('{"vehicle_id":"V001","speed":65.5,"rpm":2500}',),
                ('{"vehicle_id":"V002","speed":72.0,"rpm":2800}',),
            ],
            ["value"],
        )
        result = transformer.transform(df)
        assert result.count() == 2
        assert "ingestion_time" in result.columns

    def test_row_count(self, spark):
        transformer = BronzeTransformer()
        df = spark.createDataFrame(
            [
                ('{"vehicle_id":"V001","speed":65.5,"rpm":2500}',),
                ('{"vehicle_id":"V002","speed":72.0,"rpm":2800}',),
            ],
            ["value"],
        )
        result = transformer.transform(df)
        assert result.count() == 2
        assert "ingestion_time" in result.columns

    def test_schema(self, spark):
        transformer = BronzeTransformer()
        df = spark.createDataFrame(
            [('{"vehicle_id":"V001","speed":65.5,"rpm":2500}',)], ["value"]
        )
        result = transformer.transform(df)
        assert "vehicle_id" in result.columns
        assert "speed" in result.columns
        assert "ingestion_time" in result.columns

    def test_values(self, spark):
        transformer = BronzeTransformer()
        df = spark.createDataFrame(
            [('{"vehicle_id":"V001","speed":65.5,"rpm":2500}',)], ["value"]
        )
        result = transformer.transform(df)
        row = result.collect()[0]
        assert row["vehicle_id"] == "V001"
        assert row["speed"] == 65.5
        assert row["rpm"] == 2500

    def test_empty_dataframe(self, spark):
        transformer = BronzeTransformer()
        empty = spark.createDataFrame([], schema)
        result = transformer.transform(empty)
        assert result.count() == 0

    def test_null_values(self, spark):
        transformer = BronzeTransformer()
        df = spark.createDataFrame(
            [('{"vehicle_id":"V001","speed":null,"rpm":2500}',)], ["value"]
        )
        result = transformer.transform(df)
        assert result.count() == 1

    def test_large_dataset(self, spark):
        transformer = BronzeTransformer()
        rows = [
            (f'{{"vehicle_id":"V{i}","speed":{i}.0,"rpm":{1000+i}}}',)
            for i in range(5000)
        ]
        df = spark.createDataFrame(rows, ["value"])
        result = transformer.transform(df)
        assert result.count() == 5000

    def test_invalid_json(self, spark):
        transformer = BronzeTransformer()
        df = spark.createDataFrame([('{"vehicle_id":"V001",}',)], ["value"])
        result = transformer.transform(df)
        assert result.count() == 1
        row = result.collect()[0]
        assert row["vehicle_id"] is None

    def test_bronze_transformer(self, spark):
        transformer = BronzeTransformer()
        df = spark.createDataFrame([Row(value='{"vehicle_id":"V1","speed":60}')])
        result = transformer.transform(df)
        assert "vehicle_id" in result.columns
        assert "ingestion_time" in result.columns
        assert result.count() == 1
