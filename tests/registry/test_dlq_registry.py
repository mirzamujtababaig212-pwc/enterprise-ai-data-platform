from common.registry.dlq_registry import DLQ_REGISTRY


def test_registry_contains_delta():
    assert "delta" in DLQ_REGISTRY

def test_registry_contains_noop():
    assert "noop" in DLQ_REGISTRY

def test_registry_values_are_classes():
    for cls in DLQ_REGISTRY.values():
        assert callable(cls)
