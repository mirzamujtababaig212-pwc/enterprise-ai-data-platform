from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

PIPELINE_CONFIGS = (
    "bronze.yaml",
    "silver.yaml",
    "gold.yaml",
    "silver_streaming.yaml",
)

CONFIG_ROOT = Path("/tmp/enterprise-ai-platform")


def _parse_arguments(
    argv: Sequence[str],
) -> tuple[str, str, str, str, str, str]:
    try:
        from awsglue.utils import getResolvedOptions
    except ImportError as exc:
        raise RuntimeError(
            "AWS Glue runtime libraries are required for the Glue pipeline job."
        ) from exc

    args = getResolvedOptions(
        list(argv),
        [
            "PIPELINE_NAME",
            "MODE",
            "APP_ENV",
            "ARTIFACT_BUCKET",
            "RELEASE_VERSION",
            "RUN_ID",
        ],
    )

    pipeline_name = args["PIPELINE_NAME"].strip().lower()
    mode = args["MODE"].strip().lower()
    app_env = args["APP_ENV"].strip().lower()
    artifact_bucket = args["ARTIFACT_BUCKET"].strip()
    release_version = args["RELEASE_VERSION"].strip()
    run_id = args["RUN_ID"].strip()

    if not pipeline_name:
        raise ValueError("PIPELINE_NAME cannot be empty.")

    if mode not in {"batch", "stream"}:
        raise ValueError(f"Unsupported pipeline mode '{mode}'. " "Supported modes: batch, stream.")

    if not app_env:
        raise ValueError("APP_ENV cannot be empty.")

    if not artifact_bucket:
        raise ValueError("ARTIFACT_BUCKET cannot be empty.")

    if not release_version:
        raise ValueError("RELEASE_VERSION cannot be empty.")

    if not run_id:
        raise ValueError("RUN_ID cannot be empty.")

    return (
        pipeline_name,
        mode,
        app_env,
        artifact_bucket,
        release_version,
        run_id,
    )


def _download_configuration(
    *,
    artifact_bucket: str,
    release_version: str,
    destination: Path = CONFIG_ROOT,
) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for the Glue configuration bootstrap.") from exc

    s3 = boto3.client("s3")

    for relative_path in (
        "config/environments/aws.yaml",
        *(f"config/pipelines/{name}" for name in PIPELINE_CONFIGS),
    ):
        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        s3_key = f"glue/enterprise-ai-platform/" f"{release_version}/{relative_path}"

        s3.download_file(
            artifact_bucket,
            s3_key,
            str(destination_path),
        )


def main(argv: Sequence[str] | None = None) -> None:
    (
        pipeline_name,
        mode,
        app_env,
        artifact_bucket,
        release_version,
        run_id,
    ) = _parse_arguments(argv if argv is not None else sys.argv)

    _download_configuration(
        artifact_bucket=artifact_bucket,
        release_version=release_version,
    )

    os.environ["APP_ENV"] = app_env
    os.environ["ENTERPRISE_AI_PLATFORM_ROOT"] = str(CONFIG_ROOT)

    from common.aws.pipeline_catalog import build_aws_glue_synchronizer
    from common.config.pipeline_execution_config import PipelineExecutionConfig
    from common.execution.pipeline_execution_runtime import (
        PipelineExecutionRuntime,
    )
    from common.runner.pipeline_runner import PipelineRunner

    execution_config = PipelineExecutionConfig(runtime="glue")

    spark = PipelineExecutionRuntime.create_spark_session(
        execution_config,
        app_name=f"EnterpriseAIPlatform-{pipeline_name}",
    )

    try:
        glue_synchronizer = build_aws_glue_synchronizer()

        PipelineRunner.run(
            pipeline_name,
            spark,
            mode=mode,
            glue_synchronizer=glue_synchronizer,
            run_id=run_id,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
