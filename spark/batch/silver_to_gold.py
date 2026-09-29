from common.aws.pipeline_catalog import build_aws_glue_synchronizer
from common.runner.pipeline_runner import PipelineRunner
from common.spark.spark_builder import SparkSessionBuilder


def main():

    spark = SparkSessionBuilder.build("SilverToGold")

    try:
        glue_synchronizer = build_aws_glue_synchronizer()

        PipelineRunner.run(
            "gold",
            spark,
            mode="batch",
            glue_synchronizer=glue_synchronizer,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
