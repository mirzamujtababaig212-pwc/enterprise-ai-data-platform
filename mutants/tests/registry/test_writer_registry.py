from common.registry.writer_registry import WRITER_REGISTRY


def test_registry_contains_delta():
    assert "delta" in WRITER_REGISTRY


def test_registry_contains_postgres():
    assert "postgres" in WRITER_REGISTRY


def test_registry_contains_console():
    assert "console" in WRITER_REGISTRY


def test_registry_contains_s3():
    assert "s3" in WRITER_REGISTRY


def test_registry_contains_iceberg():
    assert "iceberg" in WRITER_REGISTRY


def test_registry_values_are_classes():
    for cls in WRITER_REGISTRY.values():
        assert callable(cls)
