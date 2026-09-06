from common.factories.pipeline_factory import PipelineFactory


def test_get_bronze_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("bronze", spark)

    assert pipeline.__class__.__name__ == "BronzePipeline"
    assert pipeline.config.pipeline_name == "bronze"


def test_get_silver_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("silver", spark)

    assert pipeline.__class__.__name__ == "SilverPipeline"
    assert pipeline.config.pipeline_name == "silver"


def test_get_gold_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("gold", spark)

    assert pipeline.__class__.__name__ == "GoldPipeline"
    assert pipeline.config.pipeline_name == "gold"


def test_unknown_pipeline_case_variant(spark):
    import pytest

    with pytest.raises(FileNotFoundError):
        PipelineFactory.get_pipeline("BRONZE", spark)


def test_unknown_pipeline():
    import pytest

    with pytest.raises((ValueError, FileNotFoundError)):
        PipelineFactory.get_pipeline("dummy", None)
