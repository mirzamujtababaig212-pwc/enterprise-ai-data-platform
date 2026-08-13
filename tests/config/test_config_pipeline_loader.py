from common.config.pipeline_loader import PipelineLoader


def test_pipeline_loader():
    result = PipelineLoader.load("bronze")

    assert result["pipeline"]["class"] == "bronze"
    assert result["reader"]["type"] == "kafka"
    assert result["writer"]["type"] == "delta"
