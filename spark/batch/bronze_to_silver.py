from common.aws.pipeline_catalog import build_aws_glue_synchronizer
from common.factories.pipeline_factory import PipelineFactory
from common.spark.spark_builder import SparkSessionBuilder


def main():
    spark = SparkSessionBuilder.build("SilverBatch")
    try:
        glue_synchronizer = build_aws_glue_synchronizer()
        pipeline = PipelineFactory.get_pipeline(
            "silver",
            spark,
            glue_synchronizer=glue_synchronizer,
        )
        pipeline.run_batch()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
