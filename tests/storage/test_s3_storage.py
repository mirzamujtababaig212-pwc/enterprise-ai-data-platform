from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from common.storage.s3_storage import S3Storage

pytestmark = pytest.mark.aws


@mock_aws
def test_s3_storage_write_read_exists_delete():
    region = "us-east-1"
    bucket = "enterprise-data-ai-platform"

    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    s3.create_bucket(
        Bucket=bucket,
    )

    storage = S3Storage(
        bucket=bucket,
        region=region,
    )

    key = "bronze/test.txt"
    payload = b"enterprise-ai-platform"

    storage.write(key, payload)

    assert storage.exists(key)

    assert storage.read(key) == payload

    assert storage.uri(key) == ("s3://enterprise-data-ai-platform/bronze/test.txt")

    storage.delete(key)

    assert not storage.exists(key)


@mock_aws
def test_s3_storage_nested_key():
    region = "us-east-1"
    bucket = "enterprise-data-ai-platform"

    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    s3.create_bucket(
        Bucket=bucket,
    )

    storage = S3Storage(
        bucket=bucket,
        region=region,
    )

    key = "silver/" "year=2026/" "month=08/" "day=15/" "data.txt"

    payload = b"partitioned-data"

    storage.write(key, payload)

    assert storage.exists(key)
    assert storage.read(key) == payload


@mock_aws
def test_s3_storage_missing_object():
    region = "us-east-1"
    bucket = "enterprise-data-ai-platform"

    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    s3.create_bucket(
        Bucket=bucket,
    )

    storage = S3Storage(
        bucket=bucket,
        region=region,
    )

    assert not storage.exists("bronze/missing.txt")


@mock_aws
def test_s3_storage_missing_read_raises():
    region = "us-east-1"
    bucket = "enterprise-data-ai-platform"

    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    s3.create_bucket(
        Bucket=bucket,
    )

    storage = S3Storage(
        bucket=bucket,
        region=region,
    )

    with pytest.raises(FileNotFoundError):
        storage.read("bronze/missing.txt")
