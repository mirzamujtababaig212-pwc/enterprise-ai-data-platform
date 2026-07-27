class MetricsBuilder:

    @staticmethod
    def build(metrics_cls, config):

        if metrics_cls is None:
            raise ValueError("Invalid Metrics class")

        return metrics_cls()
