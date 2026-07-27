from common.validation.rules.range_rule import RangeRule


def test_range(spark):
    rule = RangeRule(
        column="age",
        minimum=18,
        maximum=60,
    )
    df = spark.createDataFrame(
        [
            (25,),
            (12,),
            (70,),
        ],
        ["age"],
    )
    valid_df, invalid_df = rule.validate(df)
    assert valid_df.count() == 1
    assert invalid_df.count() == 2
