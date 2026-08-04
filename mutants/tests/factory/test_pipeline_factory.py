from common.factories.pipeline_factory import PipelineFactory


def test_create_bronze_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("bronze", spark)
    assert pipeline.config.pipeline_name == "Bronze"


def test_create_silver_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("silver", spark)
    assert pipeline.config.pipeline_name == "Silver"


def test_create_gold_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("gold", spark)
    assert pipeline.config.pipeline_name == "Gold"
