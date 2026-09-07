from common.config.settings import Settings
from common.registry.dlq_registry import DLQ_REGISTRY


class DLQFactory:

    @staticmethod
    def create(config):

        dlq_cfg = config.get("dlq", {})
        dlq_type = dlq_cfg.get("type", "default")

        pipeline = config.get("pipeline", {}).get("class")

        if dlq_type == "delta":
            dlq_cls = DLQ_REGISTRY["delta"]

            table = dlq_cfg.get("table")

            if not table:
                table_map = {
                    "bronze": Settings.storage.BRONZE_DLQ_TABLE,
                    "silver": Settings.storage.SILVER_DLQ_TABLE,
                }

                table = table_map.get(
                    pipeline,
                    f"{pipeline}.dlq",
                )

            return dlq_cls(table=table)

        if dlq_type == "noop":
            return DLQ_REGISTRY["noop"]()

        if dlq_type != "default":
            raise ValueError(f"Unknown DLQ type: {dlq_type}")

        if pipeline == "bronze":
            return DLQ_REGISTRY["delta"](table=Settings.storage.BRONZE_DLQ_TABLE)

        if pipeline == "silver":
            return DLQ_REGISTRY["delta"](table=Settings.storage.SILVER_DLQ_TABLE)

        return DLQ_REGISTRY["noop"]()
