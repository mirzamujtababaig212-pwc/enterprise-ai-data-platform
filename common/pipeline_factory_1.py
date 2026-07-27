from common.config.settings import Settings
from common.dependency_provider import DependencyProvider
from common.dlq.delta_dlq import DeltaDLQ
from common.dlq.noop_dlq import NoOpDLQ
from common.metrics.metrics_collector import MetricsCollector
from common.pipelines.bronze_pipeline import BronzePipeline
from common.pipelines.gold_pipeline import GoldPipeline
from common.pipelines.silver_pipeline import SilverPipeline
from common.validation.composite_validator import CompositeValidator
from common.validation.noop_validator import NoOpValidator


class PipelineFactory:

    @staticmethod
    def get_pipeline(name, spark):

        name = name.lower()
        metrics=MetricsCollector(),

        try:
            if name == "bronze":
                return BronzePipeline(
                        spark=spark,
                        reader=DependencyProvider.bronze_reader(),
                        validator=CompositeValidator(
                            [
                                SchemaValidator(),
                                NullValidator(
                                        ["vehicle_id", "timestamp"]
                                ),
                                DuplicateValidator(
                                        ["vehicle_id", "timestamp"]
                                )
                            ]
                        ),
                        writer=DependencyProvider.bronze_writer(),
                        transformer=DependencyProvider.bronze_transformer(),
                        metrics=metrics,
                        dlq=DeltaDLQ(
                                table=Settings.storage.BRONZE_DLQ_TABLE
                        )
                )
            elif name.lower() == "silver":
                return SilverPipeline(
                        spark=spark,
                        reader=DependencyProvider.silver_reader(),
                        validator=CompositeValidator(
                            [
                                SchemaValidator(),
                                BusinessRuleValidator(),
                                DuplicateValidator(
                                        ["vehicle_id", "timestamp"]
                                )
                            ]
                        ),
                        writer=DependencyProvider.silver_writer(),
                        transformer=DependencyProvider.silver_transformer(),
                        metrics=metrics,
                        dlq=DeltaDLQ(
                                table=Settings.storage.SILVER_DLQ_TABLE
                        )
                )
            elif name.lower() == "gold":

                return GoldPipeline(
                        spark=spark,
                        reader=DependencyProvider.gold_reader(),
                        validator=NoOpValidator(),
                        writer=DependencyProvider.gold_writer(),
                        transformer=DependencyProvider.gold_transformer(),
                        metrics=metrics,
                        dlq=NoOpDLQ()
                )
            else:
                raise ValueError(
                        f"Unknown pipeline {name}"
                )
        except KeyError:
            raise ValueError(f"Unknown pipeline: {name}")
