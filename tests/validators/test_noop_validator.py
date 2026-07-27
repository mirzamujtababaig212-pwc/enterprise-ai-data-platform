from common.validation.noop_validator import NoOpValidator


def test_returns_original_dataframe(spark):
    validator = NoOpValidator()
    df = spark.createDataFrame(
        [
            (1, "Alice"),
            (2, "Bob")
        ],
        ["id", "name"]
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == df.count()
    assert valid_df.collect() == df.collect()
    assert invalid_df is None

def test_empty_dataframe(spark):
    validator = NoOpValidator()
    df = spark.createDataFrame(
        [],
        "id INT, name STRING"
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == 0
    assert invalid_df is None

def test_large_dataset(spark):
    validator = NoOpValidator()
    rows = [
        (i, f"name{i}")
        for i in range(5000)
    ]
    df = spark.createDataFrame(
        rows,
        ["id", "name"]
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == 5000

def test_returns_same_dataframe(spark):
    validator = NoOpValidator()
    df = spark.createDataFrame(
        [
            (1, "Alice"),
            (2, "Bob"),
        ],
        ["id", "name"],
    )
    valid_df, invalid_df = validator.validate(df)
    assert valid_df.count() == df.count()
    assert invalid_df is None

def test_none_invalid_dataframe(spark):
    validator = NoOpValidator()
    df = spark.createDataFrame(
        [
            (1, "Alice"),
        ],
        ["id", "name"],
    )
    valid_df, invalid_df = validator.validate(df)
    assert invalid_df is None
