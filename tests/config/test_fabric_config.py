from common.config.fabric import FabricConfig


def test_fabric_options_returns_dictionary():

    options = FabricConfig.options()

    assert isinstance(
        options,
        dict,
    )


def test_fabric_options_excludes_empty_values():

    options = FabricConfig.options()

    for value in options.values():

        assert value is not None
        assert value != ""
