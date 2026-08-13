from unittest.mock import Mock, patch

import pytest

from common.pipeline_factory_1 import PipelineFactory


def _mock_pipeline_class():
    return Mock()


@patch("common.pipeline_factory_1.DependencyProvider")
@patch("common.pipeline_factory_1.DeltaDLQ")
@patch("common.pipeline_factory_1.MetricsCollector")
def test_get_bronze_pipeline(
    mock_metrics_class,
    mock_dlq_class,
    mock_provider_class,
    spark,
):
    mock_pipeline = _mock_pipeline_class()

    with patch(
        "common.pipeline_factory_1.BronzePipeline",
        return_value=mock_pipeline,
    ) as mock_pipeline_class:

        result = PipelineFactory.get_pipeline(
            "bronze",
            spark,
        )

    assert result is mock_pipeline
    mock_pipeline_class.assert_called_once()

    mock_provider_class.bronze_reader.assert_called_once()
    mock_provider_class.bronze_writer.assert_called_once()
    mock_provider_class.bronze_transformer.assert_called_once()


@patch("common.pipeline_factory_1.DependencyProvider")
@patch("common.pipeline_factory_1.DeltaDLQ")
@patch("common.pipeline_factory_1.MetricsCollector")
def test_get_silver_pipeline(
    mock_metrics_class,
    mock_dlq_class,
    mock_provider_class,
    spark,
):
    mock_pipeline = _mock_pipeline_class()

    with patch(
        "common.pipeline_factory_1.SilverPipeline",
        return_value=mock_pipeline,
    ) as mock_pipeline_class:

        result = PipelineFactory.get_pipeline(
            "silver",
            spark,
        )

    assert result is mock_pipeline
    mock_pipeline_class.assert_called_once()

    mock_provider_class.silver_batch_reader.assert_called_once()
    mock_provider_class.silver_writer.assert_called_once()
    mock_provider_class.silver_transformer.assert_called_once()


@patch("common.pipeline_factory_1.DependencyProvider")
@patch("common.pipeline_factory_1.DeltaDLQ")
@patch("common.pipeline_factory_1.MetricsCollector")
def test_get_gold_pipeline(
    mock_metrics_class,
    mock_dlq_class,
    mock_provider_class,
    spark,
):
    mock_pipeline = _mock_pipeline_class()

    with patch(
        "common.pipeline_factory_1.GoldPipeline",
        return_value=mock_pipeline,
    ) as mock_pipeline_class:

        result = PipelineFactory.get_pipeline(
            "gold",
            spark,
        )

    assert result is mock_pipeline
    mock_pipeline_class.assert_called_once()

    mock_provider_class.gold_reader.assert_called_once()
    mock_provider_class.gold_writer.assert_called_once()
    mock_provider_class.gold_transformer.assert_called_once()


def test_unknown_pipeline(spark):
    with pytest.raises(
        ValueError,
        match="Unknown pipeline",
    ):
        PipelineFactory.get_pipeline(
            "unknown",
            spark,
        )
