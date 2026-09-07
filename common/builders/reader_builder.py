from common.config.settings import Settings
from common.registry.reader_registry import READER_REGISTRY
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
    def build(config):

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

        if reader_type not in READER_REGISTRY:
            raise ValueError(f"Unsupported reader type: {reader_type}")

        reader_cls = READER_REGISTRY[reader_type]

        # ----------------------------------------------------------
        # Kafka
        # ----------------------------------------------------------

        if reader_type == "kafka":

            return reader_cls(Settings.kafka.options)

        # ----------------------------------------------------------
        # Snowflake
        # ----------------------------------------------------------

        if reader_type == "snowflake":

            table = cfg.get("table")

            if not table:
                raise ValueError("Table is required for Snowflake reader.")

            table = ReaderBuilder._resolve_storage_value(table)

            options = dict(Settings.snowflake.options())

            options.update(cfg.get("options", {}))

            return reader_cls(
                options=options,
                table=table,
            )

        # ----------------------------------------------------------
        # PostgreSQL / Fabric
        # ----------------------------------------------------------

        if reader_type in {
            "postgres",
            "fabric",
        }:

            table = cfg.get("table")

            if not table:
                raise ValueError(f"Table is required for reader type: " f"{reader_type}")

            table = ReaderBuilder._resolve_storage_value(table)

            return reader_cls(table=table)

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

            kwargs = {
                "path": path,
            }

            schema_name = cfg.get("schema")

            if schema_name:

                if schema_name not in SCHEMAS:
                    raise ValueError(f"Unknown schema: {schema_name}")

                kwargs["schema"] = SCHEMAS[schema_name]

            if cfg.get("table"):

                kwargs["table"] = ReaderBuilder._resolve_storage_value(cfg["table"])

            if reader_type == "csv" and "header" in cfg:

                kwargs["header"] = cfg["header"]

            return reader_cls(**kwargs)

        raise ValueError(f"Unsupported reader type: {reader_type}")
