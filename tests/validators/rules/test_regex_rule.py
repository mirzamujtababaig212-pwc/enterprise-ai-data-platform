from common.validation.rules.regex_rule import RegexRule


def test_regex(spark):
    rule = RegexRule(
        column="email",
        pattern=r".+@.+"
    )
    df = spark.createDataFrame(
        [
            ("alice@example.com",),
            ("bademail",),
        ],
        ["email"],
    )
    valid_df, invalid_df = rule.validate(df)
    assert valid_df.count() == 1
    assert invalid_df.count() == 1
