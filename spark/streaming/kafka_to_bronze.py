from common.runner.pipeline_runner import PipelineRunner
from common.spark.spark_builder import SparkSessionBuilder


def main():
    spark = SparkSessionBuilder.build(
        "KafkaToBronzeStreaming",
        include_kafka=True,
    )

    try:
        PipelineRunner.run(
            "bronze",
            spark,
            mode="stream",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
