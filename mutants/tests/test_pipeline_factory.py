from unittest.mock import Mock, patch

import pytest

from common.pipeline_factory_1 import PipelineFactory
from common.pipelines.bronze_pipeline import BronzePipeline
from common.pipelines.gold_pipeline import GoldPipeline
from common.pipelines.silver_pipeline import SilverPipeline


class TestPipelineFactory:
    @patch("common.pipeline_factory_1.DependencyProvider")
    @patch("common.pipeline_factory_1.DeltaDLQ")
    @patch("common.pipeline_factory_1.MetricsCollector")
    def test_get_bronze_pipeline(
        self,
        mock_metrics,
        mock_dlq,
        mock_provider,
    ):
        spark = Mock()

        pipeline = PipelineFactory.get_pipeline(
            "bronze",
            spark,
        )

        assert isinstance(pipeline, BronzePipeline)

        mock_provider.bronze_reader.assert_called_once()
        mock_provider.bronze_writer.assert_called_once()
        mock_provider.bronze_transformer.assert_called_once()

        mock_metrics.assert_called_once()
        mock_dlq.assert_called_once()

    @patch("common.pipeline_factory_1.DependencyProvider")
    @patch("common.pipeline_factory_1.DeltaDLQ")
    @patch("common.pipeline_factory_1.MetricsCollector")
    def test_get_silver_pipeline(
        self,
        mock_metrics,
        mock_dlq,
        mock_provider,
    ):

        spark = Mock()

        pipeline = PipelineFactory.get_pipeline(
            "silver",
            spark,
        )

        assert isinstance(pipeline, SilverPipeline)

        mock_provider.silver_reader.assert_called_once()
        mock_provider.silver_writer.assert_called_once()
        mock_provider.silver_transformer.assert_called_once()

        mock_metrics.assert_called_once()
        mock_dlq.assert_called_once()

    @patch("common.pipeline_factory_1.DependencyProvider")
    @patch("common.pipeline_factory_1.NoOpDLQ")
    @patch("common.pipeline_factory_1.MetricsCollector")
    def test_get_gold_pipeline(
        self,
        mock_metrics,
        mock_dlq,
        mock_provider,
    ):

        spark = Mock()

        pipeline = PipelineFactory.get_pipeline(
            "gold",
            spark,
        )

        assert isinstance(pipeline, GoldPipeline)

        mock_provider.gold_reader.assert_called_once()
        mock_provider.gold_writer.assert_called_once()
        mock_provider.gold_transformer.assert_called_once()

        mock_metrics.assert_called_once()
        mock_dlq.assert_called_once()

    def test_unknown_pipeline(self):
        spark = Mock()
        with pytest.raises(ValueError):
            PipelineFactory.get_pipeline(
                "dummy",
                spark,
            )

    @patch("common.pipeline_factory_1.DependencyProvider")
    @patch("common.pipeline_factory_1.DeltaDLQ")
    @patch("common.pipeline_factory_1.MetricsCollector")
    def test_pipeline_case_insensitive(
        self,
        mock_metrics,
        mock_dlq,
        mock_provider,
    ):
        spark = Mock()
        pipeline = PipelineFactory.get_pipeline(
            "BRONZE",
            spark,
        )
        assert isinstance(pipeline, BronzePipeline)

    @patch("common.pipeline_factory_1.DependencyProvider")
    def test_key_error(self, mock_provider):
        spark = Mock()
        mock_provider.bronze_reader.side_effect = KeyError
        with pytest.raises(ValueError):
            PipelineFactory.get_pipeline(
                "bronze",
                spark,
            )
