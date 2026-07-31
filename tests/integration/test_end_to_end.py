from common.factories.pipeline_factory import PipelineFactory


def test_complete_pipeline(spark):
    bronze = PipelineFactory.get_pipeline("bronze", spark)
    silver = PipelineFactory.get_pipeline("silver", spark)
    gold = PipelineFactory.get_pipeline("gold", spark)
    assert bronze is not None
    assert silver is not None
    assert gold is not None
