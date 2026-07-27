import pytest

from common.validation.composite_validator import CompositeValidator
from common.validation.duplicate_validator import DuplicateValidator
from common.validation.null_validator import NullValidator
from common.validation.schema_validator import SchemaValidator


def create_validator():
    return CompositeValidator([
        SchemaValidator(["id", "email"]),
        NullValidator(["email"]),
        DuplicateValidator(["id"]),
    ])

def test_valid_invalid_records(spark):
    validator = create_validator()
    df = spark.createDataFrame(
        [
            (1, "alice@example.com"),
            (2, None),
            (3, "bob@example.com"),
            (4, "")
        ],
        ["id", "email"]
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == 2
    assert invalid_df.count() == 2
    assert (
        valid_df.count()
        +
        invalid_df.count()
    ) == df.count()

def test_all_valid(spark):
    validator = create_validator()
    df = spark.createDataFrame(
        [
            (1, "a@abc.com"),
            (2, "b@abc.com")
        ],
        ["id", "email"]
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == 2
    assert invalid_df.count() == 0

def test_all_invalid(spark):
    validator = create_validator()
    df = spark.createDataFrame(
        [
            (1, None),
            (2, "")
        ],
        ["id", "email"]
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == 0
    assert invalid_df.count() == 2

def test_empty_dataframe(spark):
    validator = create_validator()
    df = spark.createDataFrame(
        [],
        "id INT,email STRING"
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == 0
    assert invalid_df.count() == 0

def test_large_dataset(spark):
    validator = create_validator()
    rows = []
    for i in range(10000):
        rows.append(
            (
                i,
                f"user{i}@gmail.com"
            )
        )
    df = spark.createDataFrame(
        rows,
        ["id","email"]
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count()==10000
    assert invalid_df.count()==0

def test_validate(spark):
    validator = create_validator()
    df = spark.createDataFrame(
        [
            (1, "alice@example.com"),
            (2, None),
            (3, "bob@example.com"),
            (4, ""),
        ],
        ["id", "email"],
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() + invalid_df.count() == df.count()

def test_null_values(spark):
    validator = create_validator()
    df = spark.createDataFrame(
        [
            (1, None),
            (2, "alice@example.com"),
            (3, None),
        ],
        ["id", "email"],
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == 1
    assert invalid_df.count() == 2

def test_duplicate_values(spark):
    validator = create_validator()
    df = spark.createDataFrame(
        [
            (1, "alice@example.com"),
            (1, "alice@example.com"),
            (2, "bob@example.com"),
        ],
        ["id", "email"],
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == 2
    assert invalid_df.count() == 1

def test_invalid_schema(spark):
    validator = create_validator()
    df = spark.createDataFrame(
        [
            ("Alice",),
        ],
        ["name"],
    )
    with pytest.raises(Exception):
        validator.validate(df)
