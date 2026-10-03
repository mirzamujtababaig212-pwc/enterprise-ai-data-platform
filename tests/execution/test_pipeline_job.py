import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spark.glue.pipeline_job import (
    CONFIG_ROOT,
    _download_configuration,
    _parse_arguments,
    main,
)


def test_parse_arguments_reads_glue_arguments() -> None:
    fake_utils = MagicMock()
    fake_utils.getResolvedOptions.return_value = {
        "PIPELINE_NAME": "silver",
        "MODE": "batch",
        "APP_ENV": "aws",
        "ARTIFACT_BUCKET": "enterprise-ai-platform-dev-123456789012",
        "RELEASE_VERSION": "abc123",
    }

    with patch.dict(
        sys.modules,
        {
            "awsglue": MagicMock(),
            "awsglue.utils": fake_utils,
        },
    ):
        result = _parse_arguments(
            [
                "script.py",
                "--PIPELINE_NAME",
                "silver",
                "--MODE",
                "batch",
                "--APP_ENV",
                "aws",
                "--ARTIFACT_BUCKET",
                "enterprise-ai-platform-dev-123456789012",
                "--RELEASE_VERSION",
                "abc123",
            ]
        )

    assert result == (
        "silver",
        "batch",
        "aws",
        "enterprise-ai-platform-dev-123456789012",
        "abc123",
    )

    fake_utils.getResolvedOptions.assert_called_once_with(
        [
            "script.py",
            "--PIPELINE_NAME",
            "silver",
            "--MODE",
            "batch",
            "--APP_ENV",
            "aws",
            "--ARTIFACT_BUCKET",
            "enterprise-ai-platform-dev-123456789012",
            "--RELEASE_VERSION",
            "abc123",
        ],
        [
            "PIPELINE_NAME",
            "MODE",
            "APP_ENV",
            "ARTIFACT_BUCKET",
            "RELEASE_VERSION",
        ],
    )


def test_parse_arguments_rejects_invalid_mode() -> None:
    fake_utils = MagicMock()
    fake_utils.getResolvedOptions.return_value = {
        "PIPELINE_NAME": "silver",
        "MODE": "invalid",
        "APP_ENV": "aws",
        "ARTIFACT_BUCKET": "bucket",
        "RELEASE_VERSION": "abc123",
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


def test_parse_arguments_rejects_empty_artifact_bucket() -> None:
    fake_utils = MagicMock()
    fake_utils.getResolvedOptions.return_value = {
        "PIPELINE_NAME": "silver",
        "MODE": "batch",
        "APP_ENV": "aws",
        "ARTIFACT_BUCKET": " ",
        "RELEASE_VERSION": "abc123",
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
            match="ARTIFACT_BUCKET cannot be empty",
        ):
            _parse_arguments(["script.py"])


def test_parse_arguments_rejects_empty_release_version() -> None:
    fake_utils = MagicMock()
    fake_utils.getResolvedOptions.return_value = {
        "PIPELINE_NAME": "silver",
        "MODE": "batch",
        "APP_ENV": "aws",
        "ARTIFACT_BUCKET": "bucket",
        "RELEASE_VERSION": " ",
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
            match="RELEASE_VERSION cannot be empty",
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


def test_download_configuration_uses_release_prefix(tmp_path: Path) -> None:
    fake_s3 = MagicMock()
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = fake_s3

    with patch.dict(
        sys.modules,
        {"boto3": fake_boto3},
    ):
        _download_configuration(
            artifact_bucket="enterprise-ai-platform-dev-123456789012",
            release_version="abc123",
            destination=tmp_path,
        )

    fake_boto3.client.assert_called_once_with("s3")

    expected = [
        (
            "enterprise-ai-platform-dev-123456789012",
            "glue/enterprise-ai-platform/abc123/" "config/environments/aws.yaml",
            str(tmp_path / "config/environments/aws.yaml"),
        ),
        (
            "enterprise-ai-platform-dev-123456789012",
            "glue/enterprise-ai-platform/abc123/" "config/pipelines/bronze.yaml",
            str(tmp_path / "config/pipelines/bronze.yaml"),
        ),
        (
            "enterprise-ai-platform-dev-123456789012",
            "glue/enterprise-ai-platform/abc123/" "config/pipelines/silver.yaml",
            str(tmp_path / "config/pipelines/silver.yaml"),
        ),
        (
            "enterprise-ai-platform-dev-123456789012",
            "glue/enterprise-ai-platform/abc123/" "config/pipelines/gold.yaml",
            str(tmp_path / "config/pipelines/gold.yaml"),
        ),
        (
            "enterprise-ai-platform-dev-123456789012",
            "glue/enterprise-ai-platform/abc123/" "config/pipelines/silver_streaming.yaml",
            str(tmp_path / "config/pipelines/silver_streaming.yaml"),
        ),
    ]

    assert fake_s3.download_file.call_count == len(expected)

    actual = [call.args for call in fake_s3.download_file.call_args_list]

    assert actual == expected


def test_main_sets_environment_after_configuration_bootstrap(monkeypatch) -> None:
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
        "ARTIFACT_BUCKET": "bucket",
        "RELEASE_VERSION": "abc123",
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
        monkeypatch.setenv("APP_ENV", "test")
        monkeypatch.setenv("ENTERPRISE_AI_PLATFORM_ROOT", str(CONFIG_ROOT))

        with patch("spark.glue.pipeline_job._download_configuration") as download_configuration:
            main(["script.py"])

    download_configuration.assert_called_once_with(
        artifact_bucket="bucket",
        release_version="abc123",
    )

    assert __import__("os").environ["APP_ENV"] == "aws"
    assert __import__("os").environ["ENTERPRISE_AI_PLATFORM_ROOT"] == str(CONFIG_ROOT)

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
