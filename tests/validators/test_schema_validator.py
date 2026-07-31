import pytest

from common.validation.schema_validator import SchemaValidator


def test_schema_pass(spark):
    df = spark.createDataFrame(
        [(1, "abc")],
        ["id", "name"],
    )
    validator = SchemaValidator(["id", "name"])
    valid, invalid = validator.validate(df)
    assert valid.count() == 1
    assert invalid is None


def test_schema_fail(spark):
    df = spark.createDataFrame(
        [(1,)],
        ["id"],
    )
    validator = SchemaValidator(["id", "name"])
    with pytest.raises(Exception):
        validator.validate(df)
