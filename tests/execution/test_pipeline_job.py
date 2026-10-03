import sys
from unittest.mock import MagicMock, patch

import pytest

from spark.glue.pipeline_job import _parse_arguments, main


def test_parse_arguments_reads_glue_arguments() -> None:
    fake_utils = MagicMock()
    fake_utils.getResolvedOptions.return_value = {
        "PIPELINE_NAME": "silver",
        "MODE": "batch",
        "APP_ENV": "aws",
    }

    fake_awsglue = MagicMock()
    fake_awsglue_utils = fake_utils

    with patch.dict(
        sys.modules,
        {
            "awsglue": fake_awsglue,
            "awsglue.utils": fake_awsglue_utils,
        },
    ):
        result = _parse_arguments(["script.py", "--PIPELINE_NAME", "silver", "--MODE", "batch"])

    assert result == ("silver", "batch", "aws")
    fake_utils.getResolvedOptions.assert_called_once_with(
        ["script.py", "--PIPELINE_NAME", "silver", "--MODE", "batch"],
        ["PIPELINE_NAME", "MODE", "APP_ENV"],
    )


def test_parse_arguments_rejects_invalid_mode() -> None:
    fake_utils = MagicMock()
    fake_utils.getResolvedOptions.return_value = {
        "PIPELINE_NAME": "silver",
        "MODE": "invalid",
        "APP_ENV": "aws",
    }

    with patch.dict(
        sys.modules,
        {
            "awsglue": MagicMock(),
            "awsglue.utils": fake_utils,
        },
    ):
        with pytest.raises(
            ValueError,
            match="Unsupported pipeline mode",
        ):
            _parse_arguments(["script.py"])


def test_parse_arguments_requires_glue_libraries() -> None:
    with patch.dict(
        sys.modules,
        {
            "awsglue": None,
            "awsglue.utils": None,
        },
    ):
        with pytest.raises(
            RuntimeError,
            match="AWS Glue runtime libraries are required",
        ):
            _parse_arguments(["script.py"])


def test_main_sets_environment_before_pipeline_imports(monkeypatch) -> None:
    fake_runtime_config = MagicMock()
    fake_runtime_config.return_value = MagicMock()

    fake_execution_runtime = MagicMock()
    spark = MagicMock()
    fake_execution_runtime.create_spark_session.return_value = spark

    fake_runner = MagicMock()

    fake_synchronizer_builder = MagicMock()
    glue_synchronizer = MagicMock()
    fake_synchronizer_builder.return_value = glue_synchronizer

    fake_config_module = MagicMock()
    fake_config_module.PipelineExecutionConfig = fake_runtime_config

    fake_runtime_module = MagicMock()
    fake_runtime_module.PipelineExecutionRuntime = fake_execution_runtime

    fake_catalog_module = MagicMock()
    fake_catalog_module.build_aws_glue_synchronizer = fake_synchronizer_builder

    fake_runner_module = MagicMock()
    fake_runner_module.PipelineRunner = fake_runner

    fake_awsglue_utils = MagicMock()
    fake_awsglue_utils.getResolvedOptions.return_value = {
        "PIPELINE_NAME": "silver",
        "MODE": "batch",
        "APP_ENV": "aws",
    }

    with patch.dict(
        sys.modules,
        {
            "awsglue": MagicMock(),
            "awsglue.utils": fake_awsglue_utils,
            "common.aws.pipeline_catalog": fake_catalog_module,
            "common.config.pipeline_execution_config": fake_config_module,
            "common.execution.pipeline_execution_runtime": fake_runtime_module,
            "common.runner.pipeline_runner": fake_runner_module,
        },
    ):
        monkeypatch.delenv("APP_ENV", raising=False)

        main(["script.py"])

    assert __import__("os").environ["APP_ENV"] == "aws"

    fake_runtime_config.assert_called_once_with(runtime="glue")
    fake_execution_runtime.create_spark_session.assert_called_once_with(
        fake_runtime_config.return_value,
        app_name="EnterpriseAIPlatform-silver",
    )
    fake_synchronizer_builder.assert_called_once_with()
    fake_runner.run.assert_called_once_with(
        "silver",
        spark,
        mode="batch",
        glue_synchronizer=glue_synchronizer,
    )
    spark.stop.assert_called_once()
