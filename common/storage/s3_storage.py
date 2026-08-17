from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from common.storage.base_storage import BaseStorage

BytesLike = bytes | bytearray


class S3Storage(BaseStorage):
    """
    Amazon S3 implementation of the platform storage contract.

    Parameters:
        bucket:
            S3 bucket name.

        prefix:
            Optional logical prefix added to every object key.

        region:
            AWS region used to construct the boto3 client.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
    ):
        if not bucket:
            raise ValueError("S3 bucket cannot be empty.")

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region

        self.client = boto3.client(
            "s3",
            region_name=region,
        )

    def _key(self, key: str) -> str:
        """
        Build the final S3 object key.
        """

        if not isinstance(key, str):
            raise TypeError("Storage key must be a string.")

        if not key:
            raise ValueError("Storage key cannot be empty.")

        key = key.lstrip("/")

        if self.prefix:
            return f"{self.prefix}/{key}"

        return key

    def write(self, key: str, data: BytesLike) -> None:
        """
        Write bytes to S3.
        """

        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes or bytearray.")

        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(key),
            Body=bytes(data),
        )

    def read(self, key: str) -> bytes:
        """
        Read an object from S3.
        """

        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=self._key(key),
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")

            if error_code in {
                "NoSuchKey",
                "404",
                "NotFound",
            }:
                raise FileNotFoundError(f"Storage object does not exist: {key}") from exc

            raise

        return response["Body"].read()

    def exists(self, key: str) -> bool:
        """
        Return whether an S3 object exists.
        """

        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=self._key(key),
            )
            return True

        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")

            if error_code in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return False

            raise

    def delete(self, key: str) -> None:
        """
        Delete an S3 object.

        S3 delete_object is idempotent, so deleting a missing
        object is allowed.
        """

        self.client.delete_object(
            Bucket=self.bucket,
            Key=self._key(key),
        )

    def uri(self, key: str) -> str:
        """
        Return an s3:// URI.
        """

        return f"s3://{self.bucket}/{self._key(key)}"
