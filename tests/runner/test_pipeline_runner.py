from unittest.mock import Mock

from common.runner.pipeline_runner import PipelineRunner


def test_run_forwards_glue_synchronizer_to_pipeline_factory(monkeypatch):
    spark = Mock()
    glue_synchronizer = Mock()
    pipeline = Mock()
    expected_result = object()
    pipeline.run.return_value = expected_result

    captured = {}

    def fake_get_pipeline(
        name,
        spark,
        *,
        glue_synchronizer=None,
        run_id=None,
    ):
        captured["name"] = name
        captured["spark"] = spark
        captured["glue_synchronizer"] = glue_synchronizer
        captured["run_id"] = run_id
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
        "run_id": None,
    }
    pipeline.run.assert_called_once_with(mode="batch")


def test_run_forwards_platform_run_id_to_pipeline_factory(monkeypatch):
    spark = Mock()
    pipeline = Mock()
    expected_result = object()
    pipeline.run.return_value = expected_result

    captured = {}

    def fake_get_pipeline(
        name,
        spark,
        *,
        glue_synchronizer=None,
        run_id=None,
    ):
        captured["name"] = name
        captured["spark"] = spark
        captured["glue_synchronizer"] = glue_synchronizer
        captured["run_id"] = run_id
        return pipeline

    monkeypatch.setattr(
        "common.runner.pipeline_runner.PipelineFactory.get_pipeline",
        fake_get_pipeline,
    )

    result = PipelineRunner.run(
        "bronze",
        spark,
        mode="batch",
        run_id="run-123",
    )

    assert result is expected_result
    assert captured == {
        "name": "bronze",
        "spark": spark,
        "glue_synchronizer": None,
        "run_id": "run-123",
    }
    pipeline.run.assert_called_once_with(mode="batch")
