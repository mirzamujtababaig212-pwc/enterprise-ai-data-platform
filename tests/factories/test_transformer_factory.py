import pytest

from common.factories.transformer_factory import TransformerFactory


def test_create_bronze():

    config = {"transformer": {"type": "bronze"}}

    transformer = TransformerFactory.create(config)

    assert transformer.__class__.__name__ == "BronzeTransformer"


def test_create_silver():

    config = {"transformer": {"type": "silver"}}

    transformer = TransformerFactory.create(config)

    assert transformer.__class__.__name__ == "SilverTransformer"


def test_create_gold():

    config = {"transformer": {"type": "gold"}}

    transformer = TransformerFactory.create(config)

    assert transformer.__class__.__name__ == "GoldTransformer"


def test_invalid_transformer():

    config = {"transformer": {"type": "dummy"}}

    with pytest.raises(ValueError):
        TransformerFactory.create(config)
