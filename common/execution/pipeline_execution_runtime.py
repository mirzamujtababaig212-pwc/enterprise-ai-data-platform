from __future__ import annotations

from typing import TYPE_CHECKING

from common.config.pipeline_execution_config import PipelineExecutionConfig

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class PipelineExecutionRuntime:
    """
    Creates the Spark session for the configured execution environment.

    Pipeline behavior remains owned by the existing pipeline/factory layers.
    This class only handles environment-specific Spark initialization.
    """

    @staticmethod
    def create_spark_session(
        config: PipelineExecutionConfig,
        *,
        app_name: str = "EnterpriseAIPlatform",
        include_kafka: bool = False,
    ) -> SparkSession:
        if config.runtime == "ecs":
            return PipelineExecutionRuntime._create_ecs_spark_session(
                app_name=app_name,
                include_kafka=include_kafka,
            )

        if config.runtime == "glue":
            return PipelineExecutionRuntime._create_glue_spark_session()

        raise ValueError(f"Unsupported pipeline execution runtime '{config.runtime}'.")

    @staticmethod
    def _create_ecs_spark_session(
        *,
        app_name: str,
        include_kafka: bool,
    ) -> SparkSession:
        from common.spark.spark_builder import SparkSessionBuilder

        return SparkSessionBuilder.build(
            app_name=app_name,
            include_kafka=include_kafka,
        )

    @staticmethod
    def _create_glue_spark_session() -> SparkSession:
        try:
            from awsglue.context import GlueContext
            from pyspark import SparkContext
        except ImportError as exc:
            raise RuntimeError(
                "AWS Glue runtime libraries are required for the 'glue' " "execution runtime."
            ) from exc

        spark_context = SparkContext.getOrCreate()
        glue_context = GlueContext(spark_context)

        return glue_context.spark_session
