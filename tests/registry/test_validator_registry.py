from common.registry.validator_registry import VALIDATOR_REGISTRY


def test_registry_contains_composite():
    assert "composite" in VALIDATOR_REGISTRY

def test_registry_contains_noop():
    assert "noop" in VALIDATOR_REGISTRY

def test_registry_values_are_classes():
    for cls in VALIDATOR_REGISTRY.values():
        assert callable(cls)
