from __future__ import annotations

import pytest

from common.storage.factory import create_storage
from common.storage.local_storage import LocalStorage
from common.storage.s3_storage import S3Storage


def test_factory_defaults_to_local(monkeypatch):
    monkeypatch.delenv(
        "STORAGE_BACKEND",
        raising=False,
    )

    monkeypatch.setenv(
        "LOCAL_STORAGE_PATH",
        ".storage-test",
    )

    storage = create_storage()

    assert isinstance(
        storage,
        LocalStorage,
    )


def test_factory_explicit_local(monkeypatch):
    monkeypatch.setenv(
        "LOCAL_STORAGE_PATH",
        ".storage-test",
    )

    storage = create_storage(
        backend="local",
    )

    assert isinstance(
        storage,
        LocalStorage,
    )


def test_factory_s3(monkeypatch):
    monkeypatch.setenv(
        "STORAGE_BACKEND",
        "s3",
    )

    monkeypatch.setenv(
        "S3_BUCKET",
        "enterprise-data-ai-platform",
    )

    monkeypatch.setenv(
        "AWS_REGION",
        "us-east-1",
    )

    storage = create_storage()

    assert isinstance(
        storage,
        S3Storage,
    )

    assert storage.bucket == "enterprise-data-ai-platform"
    assert storage.region == "us-east-1"


def test_factory_requires_s3_bucket(monkeypatch):
    monkeypatch.setenv(
        "STORAGE_BACKEND",
        "s3",
    )

    monkeypatch.delenv(
        "S3_BUCKET",
        raising=False,
    )

    with pytest.raises(ValueError, match="S3_BUCKET"):
        create_storage()


def test_factory_rejects_unsupported_backend():
    with pytest.raises(
        ValueError,
        match="Unsupported storage backend",
    ):
        create_storage(
            backend="unsupported",
        )
