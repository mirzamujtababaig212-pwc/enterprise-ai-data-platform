from pathlib import Path

from common.storage.local_storage import LocalStorage


def test_local_storage_write_read_exists(tmp_path: Path):
    storage = LocalStorage(base_path=tmp_path)

    key = "bronze/test.txt"
    payload = b"enterprise-ai-platform"

    storage.write(key, payload)

    assert storage.exists(key)
    assert storage.read(key) == payload


def test_local_storage_delete(tmp_path: Path):
    storage = LocalStorage(base_path=tmp_path)

    key = "bronze/delete-me.txt"

    storage.write(key, b"temporary")

    assert storage.exists(key)

    storage.delete(key)

    assert not storage.exists(key)


def test_local_storage_creates_nested_paths(tmp_path: Path):
    storage = LocalStorage(base_path=tmp_path)

    key = "bronze/year=2026/month=08/day=15/data.txt"

    storage.write(key, b"partitioned-data")

    assert storage.exists(key)
    assert storage.read(key) == b"partitioned-data"
