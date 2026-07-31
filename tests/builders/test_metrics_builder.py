import pytest

from common.builders.metrics_builder import MetricsBuilder
from common.metrics.metrics_collector import MetricsCollector


def test_build_metrics():

    metrics = MetricsBuilder.build(MetricsCollector, {})

    assert isinstance(metrics, MetricsCollector)


def test_invalid_metrics():

    with pytest.raises(ValueError):
        MetricsBuilder.build(None, {})
