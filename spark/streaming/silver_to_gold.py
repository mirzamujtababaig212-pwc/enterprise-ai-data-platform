from common.runner.pipeline_runner import PipelineRunner
from common.spark.spark_builder import SparkSessionBuilder

spark = SparkSessionBuilder.build("SilverToGold")

PipelineRunner.run("gold", spark)
