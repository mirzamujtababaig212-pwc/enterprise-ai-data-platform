import pytest

from common.factories.metrics_factory import MetricsFactory
from common.metrics.metrics_collector import MetricsCollector


def test_create_metrics():

    config = {
        "metrics": {
            "type": "default"
        }
    }

    metrics = MetricsFactory.create(config)

    assert isinstance(
        metrics,
        MetricsCollector
    )


def test_invalid_metrics():

    config = {
        "metrics": {
            "type": "dummy"
        }
    }

    with pytest.raises(ValueError):
        MetricsFactory.create(config)
