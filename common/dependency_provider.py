from common.config.settings import Settings
from common.dlq.delta_dlq import DeltaDLQ
from common.dlq.noop_dlq import NoOpDLQ
from common.metrics.metrics_collector import MetricsCollector
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader
from common.transformers.bronze_transformer import BronzeTransformer
from common.transformers.gold_transformer import GoldTransformer
from common.transformers.silver_transformer import SilverTransformer
from common.validation.composite_validator import CompositeValidator
from common.validation.noop_validator import NoOpValidator
from common.writers.delta_writer import DeltaWriter
from common.writers.postgres_writer import PostgresWriter
from spark.schemas.bronze_schema import bronze_schema
from spark.schemas.silver_schema import silver_schema


class DependencyProvider:

    @staticmethod
    def bronze_reader():
        return KafkaReader(
            Settings.kafka.options
        )

    @staticmethod
    def silver_reader():
        return ParquetReader(
            path=Settings.storage.BRONZE_PATH,
            schema=bronze_schema
        )

    @staticmethod
    def gold_reader():
        return ParquetReader(
            path=Settings.storage.SILVER_PATH,
            schema=silver_schema
        )

    @staticmethod
    def bronze_writer():
        return DeltaWriter(
            table=Settings.storage.BRONZE_TABLE,
            checkpoint=Settings.storage.BRONZE_CHECKPOINT
        )

    @staticmethod
    def silver_writer():
        return DeltaWriter(
            table=Settings.storage.SILVER_TABLE,
            checkpoint=Settings.storage.SILVER_CHECKPOINT
        )

    @staticmethod
    def gold_writer():
        return PostgresWriter(
            url=Settings.postgres.URL,
            table=Settings.postgres.TABLE,
            properties=Settings.postgres.PROPERTIES
        )

    @staticmethod
    def bronze_validator():
        return CompositeValidator(
		[
            		SchemaValidator(),
            		NullValidator(
                		["vehicle_id", "timestamp"]
            		),
            		DuplicateValidator(
                		["vehicle_id", "timestamp"]
            		)
        	]
	)

    @staticmethod
    def silver_validator():
        return CompositeValidator(
		[
        		SchemaValidator(),
        		BusinessRuleValidator(),
        		DuplicateValidator(
            			["vehicle_id", "timestamp"]
        		)
    		]
	)

    @staticmethod
    def gold_validator():
        return NoOpValidator()

    @staticmethod
    def bronze_transformer():
        return BronzeTransformer()

    @staticmethod
    def silver_transformer():
        return SilverTransformer()

    @staticmethod
    def gold_transformer():
        return GoldTransformer()

    @staticmethod
    def metrics():
        return MetricsCollector()

    @staticmethod
    def bronze_dlq():
        return DeltaDLQ(
		table=Settings.storage.BRONZE_DLQ_TABLE
	)

    @staticmethod
    def silver_dlq():
        return DeltaDLQ(
		table=Settings.storage.SILVER_DLQ_TABLE
	)

    @staticmethod
    def gold_dlq():
        return NoOpDLQ()
