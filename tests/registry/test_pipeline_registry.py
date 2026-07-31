from common.registry.pipeline_registry import PIPELINE_REGISTRY


def test_registry_contains_bronze():
    assert "bronze" in PIPELINE_REGISTRY


def test_registry_contains_silver():
    assert "silver" in PIPELINE_REGISTRY


def test_registry_contains_gold():
    assert "gold" in PIPELINE_REGISTRY


def test_registry_values_are_classes():
    for cls in PIPELINE_REGISTRY.values():
        assert callable(cls)
