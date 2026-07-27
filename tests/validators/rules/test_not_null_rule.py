from common.validation.rules.not_null_rule import NotNullRule


def test_valid_rows(spark):
    rule = NotNullRule(columns=["email"])
    df = spark.createDataFrame(
        [
            (1, "alice@example.com"),
            (2, "bob@example.com"),
        ],
        ["id", "email"],
    )
    valid_df, invalid_df = rule.validate(df)
    assert valid_df.count() == 2
    assert invalid_df.count() == 0

def test_null_rows(spark):
    rule = NotNullRule(columns=["email"])
    df = spark.createDataFrame(
        [
            (1, None),
            (2, "bob@example.com"),
        ],
        ["id", "email"],
    )
    valid_df, invalid_df = rule.validate(df)
    assert valid_df.count() == 1
    assert invalid_df.count() == 1
