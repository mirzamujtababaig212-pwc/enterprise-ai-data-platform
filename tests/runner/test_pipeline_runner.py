from unittest.mock import Mock

from common.runner.pipeline_runner import PipelineRunner


def test_run_forwards_glue_synchronizer_to_pipeline_factory(monkeypatch):
    spark = Mock()
    glue_synchronizer = Mock()
    pipeline = Mock()
    expected_result = object()
    pipeline.run.return_value = expected_result

    captured = {}

    def fake_get_pipeline(name, spark, *, glue_synchronizer=None):
        captured["name"] = name
        captured["spark"] = spark
        captured["glue_synchronizer"] = glue_synchronizer
        return pipeline

    monkeypatch.setattr(
        "common.runner.pipeline_runner.PipelineFactory.get_pipeline",
        fake_get_pipeline,
    )

    result = PipelineRunner.run(
        "gold",
        spark,
        mode="batch",
        glue_synchronizer=glue_synchronizer,
    )

    assert result is expected_result
    assert captured == {
        "name": "gold",
        "spark": spark,
        "glue_synchronizer": glue_synchronizer,
    }
    pipeline.run.assert_called_once_with(mode="batch")
