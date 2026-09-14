from __future__ import annotations

import os
import uuid

import boto3
import pytest

from common.readers.delta_reader import DeltaReader
from common.writers.delta_writer import DeltaWriter

pytestmark = pytest.mark.aws


def test_aws_delta_s3_write_read():
    if os.getenv("RUN_AWS_INTEGRATION") != "1":
        pytest.skip("Set RUN_AWS_INTEGRATION=1 to run the AWS Delta integration test")

    bucket = os.getenv(
        "AWS_DELTA_TEST_BUCKET",
        "enterprise-data-ai-platform",
    )
    region = os.getenv(
        "AWS_DEFAULT_REGION",
        "us-east-1",
    )

    test_id = uuid.uuid4().hex
    prefix = f"aws-integration-tests/delta-write-read/{test_id}"
    path = f"s3a://{bucket}/{prefix}"
    table = f"bronze.aws_delta_integration_{test_id}"

    spark = None
    s3 = boto3.client("s3", region_name=region)

    try:
        from common.spark.spark_builder import SparkSessionBuilder

        spark = SparkSessionBuilder.build(
            app_name="AWSDeltaS3IntegrationTest",
        )

        source = spark.createDataFrame(
            [
                (1, "A"),
                (2, "B"),
                (3, "C"),
            ],
            ["id", "value"],
        )

        writer = DeltaWriter(
            table=table,
            path=path,
            mode="overwrite",
        )

        writer.write_batch(source)

        assert spark.catalog.tableExists(table)

        reader = DeltaReader(path=path)
        result = reader.read(spark)

        assert result.count() == 3
        assert result.columns == ["id", "value"]
        assert sorted(result.collect()) == sorted(source.collect())

        delta_log = f"{prefix}/_delta_log/"
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=delta_log,
            MaxKeys=1,
        )

        assert response.get("KeyCount", 0) >= 1

    finally:
        if spark is not None:
            try:
                if spark.catalog.tableExists(table):
                    spark.sql(f"DROP TABLE IF EXISTS {table}")
            finally:
                spark.stop()

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=prefix,
        ):
            objects = page.get("Contents", [])
            if objects:
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={
                        "Objects": [{"Key": obj["Key"]} for obj in objects],
                    },
                )

        remaining = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=1,
        )

        assert remaining.get("KeyCount", 0) == 0
