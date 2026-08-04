from common.validation.rules.duplicate_rule import DuplicateRule


def test_duplicate_detection(spark):
    rule = DuplicateRule(["id"])
    df = spark.createDataFrame(
        [
            (1, "Alice"),
            (1, "Alice"),
            (2, "Bob"),
        ],
        ["id", "name"],
    )
    valid_df, invalid_df = rule.validate(df)
    assert valid_df.count() == 2
    assert invalid_df.count() == 1
