from common.factories.pipeline_factory import PipelineFactory
from spark.transformations.silver_to_gold_transformer import (
    SilverToGoldTransformer,
)


def test_create_bronze_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("bronze", spark)
    assert pipeline is not None
    assert pipeline.reader.__class__.__name__ == "KafkaReader"
    assert pipeline.writer.__class__.__name__ == "DeltaWriter"
    assert pipeline.transformer.__class__.__name__ == "BronzeTransformer"
    assert pipeline.validator.__class__.__name__ == "CompositeValidator"
    assert pipeline.metrics.__class__.__name__ == "MetricsCollector"
    assert pipeline.dlq.__class__.__name__ == "DeltaDLQ"


def test_create_silver_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("silver", spark)
    assert pipeline.reader.__class__.__name__ == "ParquetReader"
    assert pipeline.writer.__class__.__name__ == "DeltaWriter"
    assert pipeline.transformer.__class__.__name__ == "SilverTransformer"
    assert pipeline.validator.__class__.__name__ == "CompositeValidator"


def test_create_gold_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("gold", spark)
    assert pipeline.reader.__class__.__name__ == "DeltaReader"
    assert pipeline.writer.__class__.__name__ == "DeltaWriter"
    assert isinstance(
        pipeline.transformer,
        SilverToGoldTransformer,
    )
    assert pipeline.validator.__class__.__name__ == "NoOpValidator"
    assert pipeline.dlq.__class__.__name__ == "NoOpDLQ"
