from __future__ import annotations

import os
import sys
from typing import Sequence


def _parse_arguments(argv: Sequence[str]) -> tuple[str, str, str]:
    try:
        from awsglue.utils import getResolvedOptions
    except ImportError as exc:
        raise RuntimeError(
            "AWS Glue runtime libraries are required for the Glue pipeline job."
        ) from exc

    args = getResolvedOptions(
        list(argv),
        ["PIPELINE_NAME", "MODE", "APP_ENV"],
    )

    pipeline_name = args["PIPELINE_NAME"].strip().lower()
    mode = args["MODE"].strip().lower()
    app_env = args["APP_ENV"].strip().lower()

    if not pipeline_name:
        raise ValueError("PIPELINE_NAME cannot be empty.")

    if mode not in {"batch", "stream"}:
        raise ValueError(f"Unsupported pipeline mode '{mode}'. " "Supported modes: batch, stream.")

    if not app_env:
        raise ValueError("APP_ENV cannot be empty.")

    return pipeline_name, mode, app_env


def main(argv: Sequence[str] | None = None) -> None:
    pipeline_name, mode, app_env = _parse_arguments(argv if argv is not None else sys.argv)

    os.environ["APP_ENV"] = app_env

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
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
