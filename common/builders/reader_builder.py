from common.config.settings import Settings

from spark.schemas.bronze_schema import bronze_schema
from spark.schemas.silver_schema import silver_schema


SCHEMAS = {
    "bronze_schema": bronze_schema,
    "silver_schema": silver_schema,
}


class ReaderBuilder:

    @staticmethod
    def _resolve_storage_value(
        value,
        default=None,
    ):

        if value is None:
            return default

        if hasattr(
            Settings.storage,
            value,
        ):
            return getattr(
                Settings.storage,
                value,
            )

        return value

    @staticmethod
    def build(reader_cls, config):

        if not config:
            raise ValueError("Reader configuration cannot be empty.")

        cfg = config.get(
            "reader",
            config,
        )

        reader_type = cfg.get("type")

        if not reader_type:
            raise ValueError("Reader type is required.")

        reader_type = reader_type.lower()

        # ----------------------------------------------------------
        # Kafka
        # ----------------------------------------------------------

        if reader_type == "kafka":

            return reader_cls(Settings.kafka.options)

        # ----------------------------------------------------------
        # File / Delta readers
        # ----------------------------------------------------------

        if reader_type in {
            "parquet",
            "csv",
            "delta",
        }:

            path = ReaderBuilder._resolve_storage_value(cfg.get("path"))

            if not path:

                defaults = {
                    "parquet": Settings.storage.BRONZE_PATH,
                    "csv": Settings.storage.BRONZE_PATH,
                    "delta": Settings.storage.BRONZE_PATH,
                }

                path = defaults[reader_type]

            kwargs = {"path": path}

            schema_name = cfg.get("schema")

            if schema_name:

                if schema_name not in SCHEMAS:
                    raise ValueError(f"Unknown schema: {schema_name}")

                kwargs["schema"] = SCHEMAS[schema_name]

            # Delta readers may optionally receive
            # a table name for catalog-based reads.
            if cfg.get("table"):

                kwargs["table"] = ReaderBuilder._resolve_storage_value(cfg["table"])

            return reader_cls(**kwargs)

        raise ValueError(f"Unsupported reader type: {reader_type}")
