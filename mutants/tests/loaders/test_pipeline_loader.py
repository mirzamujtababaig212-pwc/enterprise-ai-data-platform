import pytest

from common.loaders.pipeline_loader import PipelineLoader


def test_load_bronze():
    cfg = PipelineLoader.load("bronze")
    assert cfg["pipeline"]["class"] == "bronze"


def test_load_silver():
    cfg = PipelineLoader.load("silver")
    assert cfg["pipeline"]["class"] == "silver"


def test_load_gold():
    cfg = PipelineLoader.load("gold")
    assert cfg["pipeline"]["class"] == "gold"


def test_invalid_pipeline():
    with pytest.raises(FileNotFoundError):
        PipelineLoader.load("dummy")
