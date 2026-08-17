from common.dependency_provider import (
    DependencyProvider,
)
from common.pipelines.silver_pipeline import (
    SilverPipeline,
)
from common.spark.spark_builder import (
    SparkSessionBuilder,
)


def main():

    spark = SparkSessionBuilder.build("SilverBatch")

    pipeline = SilverPipeline(
        spark=spark,
        reader=(DependencyProvider.silver_batch_reader()),
        validator=(DependencyProvider.silver_validator()),
        writer=(DependencyProvider.silver_writer()),
        transformer=(DependencyProvider.silver_transformer()),
        metrics=(DependencyProvider.metrics()),
        dlq=(DependencyProvider.silver_dlq()),
    )

    pipeline.run_batch()


if __name__ == "__main__":
    main()
