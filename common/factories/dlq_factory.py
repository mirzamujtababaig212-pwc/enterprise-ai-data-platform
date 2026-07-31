from common.registry.dlq_registry import DLQ_REGISTRY


class DLQFactory:
    @staticmethod
    def create(config):
        dlq_type = config["dlq"]["type"]
        if dlq_type not in DLQ_REGISTRY:
            raise ValueError(f"Unknown DLQ type: {dlq_type}")
        dlq_cls = DLQ_REGISTRY[dlq_type]
        if dlq_type == "delta":
            return dlq_cls(table=config["dlq"].get("table", "fallback_dlq_table"))
        return dlq_cls()
