from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from common.storage.s3_storage import S3Storage


@pytest.fixture
def s3_storage():
    storage = S3Storage(
        bucket="enterprise-data-ai-platform",
        prefix="",
        region="us-east-1",
    )

    storage.client = MagicMock()

    return storage


def test_s3_storage_write(s3_storage):
    payload = b"enterprise-ai-platform"

    s3_storage.write("raw/test.txt", payload)

    s3_storage.client.put_object.assert_called_once_with(
        Bucket="enterprise-data-ai-platform",
        Key="raw/test.txt",
        Body=payload,
    )


def test_s3_storage_read(s3_storage):
    payload = b"enterprise-ai-platform"

    body = MagicMock()
    body.read.return_value = payload

    s3_storage.client.get_object.return_value = {
        "Body": body,
    }

    result = s3_storage.read("raw/test.txt")

    assert result == payload

    s3_storage.client.get_object.assert_called_once_with(
        Bucket="enterprise-data-ai-platform",
        Key="raw/test.txt",
    )


def test_s3_storage_exists_true(s3_storage):
    s3_storage.client.head_object.return_value = {}

    assert s3_storage.exists("raw/test.txt") is True

    s3_storage.client.head_object.assert_called_once_with(
        Bucket="enterprise-data-ai-platform",
        Key="raw/test.txt",
    )


def test_s3_storage_exists_false(s3_storage):
    s3_storage.client.head_object.side_effect = ClientError(
        {
            "Error": {
                "Code": "404",
                "Message": "Not Found",
            }
        },
        "HeadObject",
    )

    assert s3_storage.exists("raw/test.txt") is False


def test_s3_storage_read_missing_object(s3_storage):
    s3_storage.client.get_object.side_effect = ClientError(
        {
            "Error": {
                "Code": "NoSuchKey",
                "Message": "The specified key does not exist.",
            }
        },
        "GetObject",
    )

    with pytest.raises(FileNotFoundError):
        s3_storage.read("raw/missing.txt")


def test_s3_storage_delete(s3_storage):
    s3_storage.delete("raw/test.txt")

    s3_storage.client.delete_object.assert_called_once_with(
        Bucket="enterprise-data-ai-platform",
        Key="raw/test.txt",
    )


def test_s3_storage_uri(s3_storage):
    assert s3_storage.uri("raw/test.txt") == "s3://enterprise-data-ai-platform/raw/test.txt"


def test_s3_storage_prefix():
    storage = S3Storage(
        bucket="enterprise-data-ai-platform",
        prefix="enterprise",
        region="us-east-1",
    )

    storage.client = MagicMock()

    storage.write(
        "raw/test.txt",
        b"hello",
    )

    storage.client.put_object.assert_called_once_with(
        Bucket="enterprise-data-ai-platform",
        Key="enterprise/raw/test.txt",
        Body=b"hello",
    )

    assert storage.uri("raw/test.txt") == "s3://enterprise-data-ai-platform/enterprise/raw/test.txt"
