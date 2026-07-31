from common.registry.metrics_registry import METRICS_REGISTRY


class MetricsFactory:
    @staticmethod
    def create(config):
        metrics_type = config["metrics"]["type"]
        if metrics_type not in METRICS_REGISTRY:
            raise ValueError(f"Unknown metrics type: {metrics_type}")
        return METRICS_REGISTRY[metrics_type]()
