from common.registry.dlq_registry import DLQ_REGISTRY


class DLQFactory:

    @staticmethod
    def create(config):

        dlq_cfg = config.get("dlq", {})
        dlq_type = dlq_cfg.get("type", "default")

        pipeline = config.get("pipeline", {}).get("class")

        # ---------------------------------------------------------
        # Explicit DLQ configuration
        # ---------------------------------------------------------

        if dlq_type == "delta":
            dlq_cls = DLQ_REGISTRY["delta"]

            return dlq_cls(table=dlq_cfg.get("table", f"{pipeline}.dlq"))

        if dlq_type == "noop":
            return DLQ_REGISTRY["noop"]()

        if dlq_type != "default":
            raise ValueError(f"Unknown DLQ type: {dlq_type}")

        # ---------------------------------------------------------
        # Pipeline-specific default DLQ
        # ---------------------------------------------------------

        if pipeline in {"bronze", "silver"}:
            return DLQ_REGISTRY["delta"](table=dlq_cfg.get("table", f"{pipeline}.dlq"))

        if pipeline == "gold":
            return DLQ_REGISTRY["noop"]()

        return DLQ_REGISTRY["noop"]()
