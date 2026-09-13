from common.runner.pipeline_runner import PipelineRunner
from common.spark.spark_builder import SparkSessionBuilder


def main() -> None:
    spark = SparkSessionBuilder.build("SilverToGoldPipeline")

    try:
        PipelineRunner.run(
            "gold",
            spark,
            mode="batch",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
