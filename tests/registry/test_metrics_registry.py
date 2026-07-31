from common.registry.metrics_registry import METRICS_REGISTRY


def test_registry_contains_default():
    assert "default" in METRICS_REGISTRY


def test_registry_values_are_classes():
    for cls in METRICS_REGISTRY.values():
        assert callable(cls)
