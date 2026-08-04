import pytest

from common.builders.transformer_builder import TransformerBuilder
from common.transformers.bronze_transformer import BronzeTransformer
from common.transformers.gold_transformer import GoldTransformer
from common.transformers.silver_transformer import SilverTransformer


def test_build_bronze():

    config = {"transformer": {"type": "bronze"}}

    transformer = TransformerBuilder.build(BronzeTransformer, config)

    assert isinstance(transformer, BronzeTransformer)


def test_build_silver():

    config = {"transformer": {"type": "silver"}}

    transformer = TransformerBuilder.build(SilverTransformer, config)

    assert isinstance(transformer, SilverTransformer)


def test_build_gold():

    config = {"transformer": {"type": "gold"}}

    transformer = TransformerBuilder.build(GoldTransformer, config)

    assert isinstance(transformer, GoldTransformer)


def test_invalid_transformer():

    with pytest.raises(ValueError):
        TransformerBuilder.build(None, {})
