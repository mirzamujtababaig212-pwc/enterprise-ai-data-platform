import sys
from unittest.mock import MagicMock, patch

import pytest

from common.config.pipeline_execution_config import PipelineExecutionConfig
from common.execution.pipeline_execution_runtime import PipelineExecutionRuntime


def test_ecs_runtime_delegates_to_spark_session_builder() -> None:
    config = PipelineExecutionConfig(runtime="ecs")
    spark = MagicMock()

    with patch(
        "common.spark.spark_builder.SparkSessionBuilder.build",
        return_value=spark,
    ) as build:
        result = PipelineExecutionRuntime.create_spark_session(
            config,
            app_name="TestECS",
            include_kafka=True,
        )

    assert result is spark
    build.assert_called_once_with(
        app_name="TestECS",
        include_kafka=True,
    )


def test_glue_runtime_creates_spark_session_from_glue_context() -> None:
    config = PipelineExecutionConfig(runtime="glue")
    spark_context = MagicMock()
    glue_context = MagicMock()
    spark = MagicMock()

    glue_context.spark_session = spark

    fake_awsglue = MagicMock()
    fake_awsglue_context = MagicMock()
    fake_awsglue_context.GlueContext.return_value = glue_context

    with patch.dict(
        sys.modules,
        {
            "awsglue": fake_awsglue,
            "awsglue.context": fake_awsglue_context,
        },
    ):
        with patch(
            "pyspark.SparkContext.getOrCreate",
            return_value=spark_context,
        ):
            result = PipelineExecutionRuntime.create_spark_session(config)

    assert result is spark
    fake_awsglue_context.GlueContext.assert_called_once_with(spark_context)


def test_glue_runtime_requires_glue_libraries() -> None:
    config = PipelineExecutionConfig(runtime="glue")

    with patch.dict(
        sys.modules,
        {
            "awsglue": None,
            "awsglue.context": None,
        },
    ):
        with pytest.raises(
            RuntimeError,
            match="AWS Glue runtime libraries are required",
        ):
            PipelineExecutionRuntime.create_spark_session(config)


def test_execution_runtime_rejects_unknown_runtime() -> None:
    config = object.__new__(PipelineExecutionConfig)
    object.__setattr__(config, "runtime", "unknown")

    with pytest.raises(
        ValueError,
        match="Unsupported pipeline execution runtime",
    ):
        PipelineExecutionRuntime.create_spark_session(config)
