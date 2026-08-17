from __future__ import annotations

import os

from common.storage.base_storage import BaseStorage
from common.storage.local_storage import LocalStorage
from common.storage.s3_storage import S3Storage


def create_storage(
    backend: str | None = None,
) -> BaseStorage:
    """
    Create the configured storage backend.

    Supported backends:

        local
        s3
    """

    backend = (backend or os.getenv("STORAGE_BACKEND") or "local").strip().lower()

    if backend == "local":
        base_path = os.getenv(
            "LOCAL_STORAGE_PATH",
            ".storage",
        )

        return LocalStorage(
            base_path=base_path,
        )

    if backend == "s3":
        bucket = os.getenv("S3_BUCKET")

        if not bucket:
            raise ValueError("S3_BUCKET must be configured when STORAGE_BACKEND=s3.")

        region = os.getenv(
            "AWS_REGION",
            os.getenv(
                "AWS_DEFAULT_REGION",
                "us-east-1",
            ),
        )

        prefix = os.getenv(
            "S3_PREFIX",
            "",
        )

        return S3Storage(
            bucket=bucket,
            prefix=prefix,
            region=region,
        )

    raise ValueError(f"Unsupported storage backend: {backend}")
