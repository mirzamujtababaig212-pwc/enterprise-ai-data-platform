from common.factories.pipeline_factory import PipelineFactory


def test_restart_pipeline(spark):
    pipeline = PipelineFactory.get_pipeline("bronze", spark)
    assert pipeline is not None
    pipeline = PipelineFactory.get_pipeline("bronze", spark)
    assert pipeline is not None
