from common.registry.transformer_registry import TRANSFORMER_REGISTRY


def test_registry_contains_bronze():
    assert "bronze" in TRANSFORMER_REGISTRY

def test_registry_contains_silver():
    assert "silver" in TRANSFORMER_REGISTRY

def test_registry_contains_gold():
    assert "gold" in TRANSFORMER_REGISTRY

def test_registry_values_are_classes():
    for cls in TRANSFORMER_REGISTRY.values():
        assert callable(cls)
