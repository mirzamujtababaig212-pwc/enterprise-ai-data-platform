from common.runner.pipeline_runner import PipelineRunner
from common.spark.spark_builder import SparkSessionBuilder


def main():

    spark = SparkSessionBuilder.build("SilverToGold")

    try:
        PipelineRunner.run(
            "gold",
            spark,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
