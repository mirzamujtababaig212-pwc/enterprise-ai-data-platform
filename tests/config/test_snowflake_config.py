from common.config.snowflake import SnowflakeConfig


def test_snowflake_options_returns_dictionary():

    options = SnowflakeConfig.options()

    assert isinstance(
        options,
        dict,
    )


def test_snowflake_options_excludes_none_values():

    options = SnowflakeConfig.options()

    for value in options.values():

        assert value is not None
        assert value != ""
