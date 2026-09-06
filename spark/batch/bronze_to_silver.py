from common.factories.pipeline_factory import PipelineFactory
from common.spark.spark_builder import SparkSessionBuilder


def main():
    spark = SparkSessionBuilder.build("SilverBatch")
    try:
        pipeline = PipelineFactory.get_pipeline("silver", spark)
        pipeline.run_batch()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
