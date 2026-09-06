from common.factories.pipeline_factory import PipelineFactory
from common.spark.spark_builder import SparkSessionBuilder


def main():
    spark = SparkSessionBuilder.build("SilverStreaming")
    try:
        pipeline = PipelineFactory.get_pipeline("silver_streaming", spark)
        pipeline.run_stream()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
