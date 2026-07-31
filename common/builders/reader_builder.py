from common.config.settings import Settings
from spark.schemas.bronze_schema import bronze_schema
from spark.schemas.silver_schema import silver_schema

SCHEMAS = {
    "bronze_schema": bronze_schema,
    "silver_schema": silver_schema,
}


class ReaderBuilder:
    @staticmethod
    def build(reader_cls, config):
        cfg = config["reader"]
        reader_type = cfg["type"]
        kwargs = {}
        if reader_type == "parquet" or reader_type == "csv" or reader_type == "delta":
            kwargs["path"] = getattr(Settings.storage, cfg.get("path", "BRONZE_PATH"))
        elif reader_type == "kafka":
            return reader_cls(Settings.kafka.options)
        if "schema" in cfg:
            kwargs["schema"] = SCHEMAS[cfg["schema"]]
        return reader_cls(**kwargs)
