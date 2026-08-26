from common.config.settings import Settings
from common.dlq.delta_dlq import DeltaDLQ
from common.dlq.noop_dlq import NoOpDLQ
from common.metrics.metrics_collector import (
    MetricsCollector,
)
from common.readers.delta_reader import DeltaReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader
from common.transformers.bronze_transformer import (
    BronzeTransformer,
)
from common.transformers.silver_transformer import (
    SilverTransformer,
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
from common.validation.noop_validator import (
    NoOpValidator,
)
from common.validation.null_validator import (
    NullValidator,
)
from common.validation.schema_validator import (
    SchemaValidator,
)
from common.writers.delta_writer import DeltaWriter
from common.writers.postgres_writer import (
    PostgresWriter,
)
from spark.schemas.bronze_schema import (
    bronze_schema,
)
from spark.schemas.silver_schema import (
    silver_schema,
)


class DependencyProvider:

    # =========================================================
    # Readers
    # =========================================================

    @staticmethod
    def bronze_reader():

        return KafkaReader(Settings.kafka.options)

    @staticmethod
    def silver_batch_reader():

        return ParquetReader(
            path=Settings.storage.BATCH_BRONZE_PATH,
            schema=bronze_schema,
        )

    @staticmethod
    def silver_stream_reader():

        return DeltaReader(
            path=Settings.storage.BRONZE_PATH,
            schema=bronze_schema,
        )

    @staticmethod
    def gold_reader():

        return DeltaReader(
            path=Settings.storage.SILVER_PATH,
            schema=silver_schema,
        )

    # =========================================================
    # Writers
    # =========================================================

    @staticmethod
    def bronze_writer():

        return DeltaWriter(
            table=Settings.storage.BRONZE_TABLE,
            path=Settings.storage.BRONZE_PATH,
            checkpoint=(Settings.storage.BRONZE_CHECKPOINT),
            mode="append",
        )

    @staticmethod
    def silver_writer():

        return DeltaWriter(
            table=Settings.storage.SILVER_TABLE,
            path=Settings.storage.SILVER_PATH,
            checkpoint=(Settings.storage.SILVER_CHECKPOINT),
            mode="append",
        )

    @staticmethod
    def gold_writer():

        return PostgresWriter(
            url=Settings.postgres.URL,
            table=Settings.postgres.TABLE,
            properties=Settings.postgres.PROPERTIES,
        )

    # =========================================================
    # Validators
    # =========================================================

    @staticmethod
    def bronze_validator():

        return CompositeValidator(
            [
                SchemaValidator(
                    [
                        "kafka_key",
                        "kafka_topic",
                        "kafka_partition",
                        "kafka_offset",
                        "kafka_timestamp",
                        "raw_value",
                        "vehicle_id",
                        "event_time",
                        "latitude",
                        "longitude",
                        "speed",
                        "rpm",
                        "fuel_level",
                        "battery",
                        "engine_temperature",
                        "gear",
                        "ingestion_time",
                    ]
                ),
                NullValidator(
                    [
                        "vehicle_id",
                        "event_time",
                    ]
                ),
                DuplicateValidator(
                    [
                        "vehicle_id",
                        "event_time",
                    ]
                ),
            ]
        )

    @staticmethod
    def silver_validator():

        return CompositeValidator(
            [
                SchemaValidator(
                    [
                        "vehicle_id",
                        "event_time",
                        "latitude",
                        "longitude",
                        "speed",
                        "rpm",
                        "fuel_level",
                        "battery",
                        "engine_temperature",
                        "gear",
                        "topic",
                        "partition",
                        "offset",
                        "timestamp",
                        "ingestion_timestamp",
                        "speed_category",
                        "fuel_status",
                        "battery_status",
                        "vehicle_status",
                    ]
                ),
                BusinessRuleValidator(),
                DuplicateValidator(
                    [
                        "vehicle_id",
                        "timestamp",
                    ]
                ),
            ]
        )

    @staticmethod
    def gold_validator():

        return NoOpValidator()

    # =========================================================
    # Transformers
    # =========================================================

    @staticmethod
    def bronze_transformer():

        return BronzeTransformer()

    @staticmethod
    def silver_transformer():

        return SilverTransformer()

    # =========================================================
    # Metrics
    # =========================================================

    @staticmethod
    def metrics():

        return MetricsCollector()

    # =========================================================
    # DLQ
    # =========================================================

    @staticmethod
    def bronze_dlq():

        return DeltaDLQ(table=Settings.storage.BRONZE_DLQ_TABLE)

    @staticmethod
    def silver_dlq():

        return DeltaDLQ(table=Settings.storage.SILVER_DLQ_TABLE)

    @staticmethod
    def gold_dlq():

        return NoOpDLQ()
