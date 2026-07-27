from common.runner.pipeline_runner import PipelineRunner
from common.spark.spark_builder import SparkSessionBuilder

spark = SparkSessionBuilder.build()

PipelineRunner.run(
        "bronze",
        spark
)

