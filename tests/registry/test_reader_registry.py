from common.registry.reader_registry import READER_REGISTRY


def test_registry_contains_kafka():
    assert "kafka" in READER_REGISTRY

def test_registry_contains_parquet():
    assert "parquet" in READER_REGISTRY

def test_registry_contains_csv():
    assert "csv" in READER_REGISTRY

def test_registry_contains_delta():
    assert "delta" in READER_REGISTRY

def test_registry_values_are_classes():
    for cls in READER_REGISTRY.values():
        assert callable(cls)
