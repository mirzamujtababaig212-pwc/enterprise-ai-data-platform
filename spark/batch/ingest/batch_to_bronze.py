from common.config.settings import Settings
from common.pipelines.base_pipeline import (
    BasePipeline,
)
from common.readers.parquet_reader import (
    ParquetReader,
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

    CONFIG = type(
        "Config",
        (),
        {
            "pipeline_name": "BatchToBronze",
            "source": "parquet",
            "path": Settings.storage.BATCH_INPUT_PATH,
            "target": "delta",
            "table": Settings.storage.BRONZE_TABLE,
            "checkpoint": "",
            "query_name": None,
            "output_mode": "append",
            "trigger": None,
            "retries": 3,
            "retry_delay": 2,
            "enable_validation": True,
            "enable_metrics": False,
            "enable_dlq": False,
            "execution_mode": "batch",
        },
    )()

    def __init__(self, spark):

        reader = ParquetReader(path=Settings.storage.BATCH_INPUT_PATH)

        transformer = BatchBronzeTransformer()

        validator = CompositeValidator(
            [
                DuplicateValidator(
                    keys=[
                        "vehicle_id",
                        "event_timestamp",
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

        super().__init__(
            spark=spark,
            reader=reader,
            validator=validator,
            writer=writer,
            transformer=transformer,
            metrics=None,
            dlq=None,
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
