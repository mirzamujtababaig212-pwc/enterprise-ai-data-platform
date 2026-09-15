from common.config.settings import Settings
from common.pipelines.base_pipeline import (
    BasePipeline,
)
from common.pipelines.pipeline_runtime_config import (
    PipelineRuntimeConfig,
)
from common.readers.csv_reader import (
    CSVReader,
)
from spark.schemas.bronze_schema import (
    bronze_schema,
)
from common.transformers.batch_bronze_transformer import (
    BatchBronzeTransformer,
)
from common.validation.business_rule_validator import (
    BusinessRuleValidator,
)
from common.validation.composite_validator import (
    CompositeValidator,
)
from common.validation.duplicate_validator import (
    DuplicateValidator,
)
from common.writers.delta_writer import (
    DeltaWriter,
)


class BatchToBronzePipeline(BasePipeline):

    def __init__(self, spark):

        reader = CSVReader(
            path=Settings.storage.RAW_VEHICLE_DATA_PATH,
            schema=bronze_schema,
        )

        transformer = BatchBronzeTransformer()

        validator = CompositeValidator(
            [
                DuplicateValidator(
                    keys=[
                        "vehicle_id",
                        "event_time",
                    ]
                ),
                BusinessRuleValidator(),
            ]
        )

        writer = DeltaWriter(
            table=Settings.storage.BRONZE_TABLE,
            path=Settings.storage.BRONZE_PATH,
            mode="append",
        )

        config = PipelineRuntimeConfig(
            pipeline_name="BatchToBronze",
            output_mode="append",
            retries=3,
            retry_delay=2,
            enable_validation=True,
            enable_metrics=False,
            enable_dlq=False,
        )

        super().__init__(
            spark=spark,
            reader=reader,
            validator=validator,
            writer=writer,
            transformer=transformer,
            metrics=None,
            dlq=None,
            config=config,
        )


def main():

    from common.spark.spark_builder import (
        SparkSessionBuilder,
    )

    spark = SparkSessionBuilder.build("BatchToBronze")

    pipeline = BatchToBronzePipeline(spark)

    pipeline.run_batch()


if __name__ == "__main__":
    main()
