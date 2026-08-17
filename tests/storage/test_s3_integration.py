from __future__ import annotations

import os
import uuid

import pytest

from common.storage.s3_storage import S3Storage


@pytest.mark.aws
@pytest.mark.skipif(
    not os.getenv("TEST_S3_BUCKET"),
    reason="TEST_S3_BUCKET is not configured",
)
def test_real_s3_round_trip():
    bucket = os.environ["TEST_S3_BUCKET"]

    region = os.getenv(
        "AWS_DEFAULT_REGION",
        "us-east-1",
    )

    storage = S3Storage(
        bucket=bucket,
        region=region,
    )

    key = "checkpoints/" f"integration-test-{uuid.uuid4().hex}.txt"

    payload = b"enterprise-ai-platform-real-s3-test"

    try:
        storage.write(
            key,
            payload,
        )

        assert storage.exists(key)

        assert storage.read(key) == payload

        assert storage.uri(key) == (f"s3://{bucket}/{key}")

    finally:
        storage.delete(key)

    assert not storage.exists(key)
