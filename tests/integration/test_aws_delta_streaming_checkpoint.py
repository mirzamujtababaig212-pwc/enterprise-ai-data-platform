from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import boto3
import pytest

from common.readers.delta_reader import DeltaReader
from common.spark.spark_builder import SparkSessionBuilder
from common.writers.delta_writer import DeltaWriter

pytestmark = pytest.mark.aws


def test_aws_delta_streaming_checkpoint_recovery():
    if os.getenv("RUN_AWS_INTEGRATION") != "1":
        pytest.skip("Set RUN_AWS_INTEGRATION=1 to run the AWS integration test")

    bucket = os.getenv(
        "AWS_DELTA_TEST_BUCKET",
        "enterprise-data-ai-platform",
    )
    region = os.getenv(
        "AWS_DEFAULT_REGION",
        "us-east-1",
    )

    test_id = uuid.uuid4().hex
    prefix = f"aws-integration-tests/delta-streaming/{test_id}"
    delta_prefix = f"{prefix}/delta"
    checkpoint_prefix = f"{prefix}/checkpoint"

    delta_path = f"s3a://{bucket}/{delta_prefix}"
    checkpoint_path = f"s3a://{bucket}/{checkpoint_prefix}"

    table = f"bronze.aws_delta_streaming_{test_id}"

    input_dir = Path(tempfile.mkdtemp(prefix="aws_delta_streaming_"))

    spark = None
    s3 = boto3.client("s3", region_name=region)

    def write_input_file(filename: str, rows: list[tuple[int, str]]) -> None:
        path = input_dir / filename

        lines = ["id,value"]
        lines.extend(f"{row_id},{value}" for row_id, value in rows)

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def foreach_batch(batch_df, batch_id):
        if batch_df.isEmpty():
            return

        DeltaWriter(
            table=table,
            path=delta_path,
            mode="append",
        ).write_batch(batch_df)

    try:
        spark = SparkSessionBuilder.build(
            app_name="AWSDeltaStreamingCheckpointTest",
        )

        write_input_file(
            "batch-1.csv",
            [
                (1, "A"),
                (2, "B"),
                (3, "C"),
            ],
        )

        source = (
            spark.readStream.format("csv")
            .option("header", "true")
            .schema("id INT, value STRING")
            .load(str(input_dir))
        )

        writer = DeltaWriter(
            table=table,
            path=delta_path,
            mode="append",
        )

        query = writer.write_stream(
            df=source,
            foreach_batch=foreach_batch,
            checkpoint=checkpoint_path,
            output_mode="append",
            query_name=f"aws_delta_streaming_{test_id}",
            trigger={"availableNow": True},
        )

        query.awaitTermination()

        assert query.exception() is None

        checkpoint_objects = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=checkpoint_prefix,
            MaxKeys=1,
        )

        assert checkpoint_objects.get("KeyCount", 0) >= 1

        first_result = DeltaReader(path=delta_path).read(spark)

        assert first_result.count() == 3
        assert {(row.id, row.value) for row in first_result.collect()} == {
            (1, "A"),
            (2, "B"),
            (3, "C"),
        }

        write_input_file(
            "batch-2.csv",
            [
                (4, "D"),
                (5, "E"),
            ],
        )

        source_restarted = (
            spark.readStream.format("csv")
            .option("header", "true")
            .schema("id INT, value STRING")
            .load(str(input_dir))
        )

        query_restarted = writer.write_stream(
            df=source_restarted,
            foreach_batch=foreach_batch,
            checkpoint=checkpoint_path,
            output_mode="append",
            query_name=f"aws_delta_streaming_{test_id}",
            trigger={"availableNow": True},
        )

        query_restarted.awaitTermination()

        assert query_restarted.exception() is None

        final_result = DeltaReader(path=delta_path).read(spark)

        assert final_result.count() == 5
        assert {(row.id, row.value) for row in final_result.collect()} == {
            (1, "A"),
            (2, "B"),
            (3, "C"),
            (4, "D"),
            (5, "E"),
        }

    finally:
        if spark is not None:
            try:
                if spark.catalog.tableExists(table):
                    spark.sql(f"DROP TABLE IF EXISTS {table}")
            except Exception:
                pass

            spark.stop()

        shutil.rmtree(input_dir, ignore_errors=True)

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
