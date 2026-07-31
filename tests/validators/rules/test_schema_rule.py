from pyspark.sql.types import *

from common.validation.rules.schema_rule import SchemaRule


def test_schema_matches(spark):
    expected = StructType(
        [
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ]
    )
    df = spark.createDataFrame([(1, "Alice")], expected)
    rule = SchemaRule(expected)
    assert rule.validate(df)
