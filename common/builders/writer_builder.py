from __future__ import annotations

from common.config.settings import Settings


class WriterBuilder:
    """
    Canonical constructor for concrete writer implementations.

    The registry determines which concrete writer class should be used.
    This builder determines how configuration is translated into the
    constructor arguments expected by that writer.
    """

    @staticmethod
    def _resolve_storage_value(
        value,
        default=None,
    ):
        """
        Resolve symbolic StorageConfig values.

        Examples:

            BRONZE_PATH
            SILVER_PATH
            GOLD_PATH
            BRONZE_CHECKPOINT

        Literal values are returned unchanged.
        """

        if value is None:
            return default

        value = str(value)

        if hasattr(Settings.storage, value):
            return getattr(Settings.storage, value)

        return value

    @staticmethod
    def build(writer_cls, config):
        """
        Construct a writer using the canonical writer configuration.

        Parameters
        ----------
        writer_cls:
            Concrete writer class obtained from WRITER_REGISTRY.

        config:
            Either:

                {"writer": {...}}

            or:

                {...}
        """

        if not config:
            raise ValueError("Writer configuration cannot be empty.")

        cfg = config.get("writer", config)

        if not isinstance(cfg, dict):
            raise ValueError("Writer configuration must be a mapping.")

        writer_type = cfg.get("type")

        if not writer_type:
            raise ValueError("Writer type is required.")

        writer_type = str(writer_type).lower().strip()

        if writer_cls is None:
            raise ValueError(f"Writer class is required for writer type: {writer_type}")

        # ==========================================================
        # DELTA
        # ==========================================================

        if writer_type == "delta":
            table = WriterBuilder._resolve_storage_value(
                cfg.get("table"),
                Settings.storage.SILVER_TABLE,
            )

            path = WriterBuilder._resolve_storage_value(cfg.get("path"))

            checkpoint = WriterBuilder._resolve_storage_value(cfg.get("checkpoint"))

            mode = cfg.get(
                "mode",
                "append",
            )

            output_mode = cfg.get(
                "output_mode",
                "append",
            )

            return writer_cls(
                table=table,
                path=path,
                mode=mode,
                checkpoint=checkpoint,
                output_mode=output_mode,
            )

        # ==========================================================
        # FABRIC
        # ==========================================================

        if writer_type == "fabric":
            table = cfg.get(
                "table",
                "",
            )

            checkpoint = WriterBuilder._resolve_storage_value(cfg.get("checkpoint"))

            mode = cfg.get(
                "mode",
                "append",
            )

            output_mode = cfg.get(
                "output_mode",
                "append",
            )

            return writer_cls(
                table=table,
                checkpoint=checkpoint,
                mode=mode,
                output_mode=output_mode,
            )

        # ==========================================================
        # PARQUET
        # ==========================================================

        if writer_type == "parquet":
            path = WriterBuilder._resolve_storage_value(cfg.get("path"))

            if not path:
                raise ValueError("Parquet writer requires a path.")

            mode = cfg.get(
                "mode",
                "append",
            )

            return writer_cls(
                path=path,
                mode=mode,
            )

        # ==========================================================
        # POSTGRES
        # ==========================================================

        if writer_type == "postgres":
            postgres = Settings.postgres

            url = cfg.get(
                "url",
                getattr(postgres, "URL", None),
            )

            table = cfg.get(
                "table",
                getattr(postgres, "TABLE", None),
            )

            properties = cfg.get(
                "properties",
                getattr(postgres, "PROPERTIES", None),
            )

            mode = cfg.get(
                "mode",
                "append",
            )

            return writer_cls(
                url=url,
                table=table,
                properties=properties,
                mode=mode,
            )

        # ==========================================================
        # SNOWFLAKE
        # ==========================================================

        if writer_type == "snowflake":
            options = cfg.get(
                "options",
                {},
            )

            table = cfg.get(
                "table",
                "",
            )

            mode = cfg.get(
                "mode",
                "append",
            )

            return writer_cls(
                options=options,
                table=table,
                mode=mode,
            )

        # ==========================================================
        # S3
        # ==========================================================

        if writer_type == "s3":
            path = WriterBuilder._resolve_storage_value(
                cfg.get("path"),
                Settings.storage.BRONZE_PATH,
            )

            return writer_cls(
                path=path,
            )

        # ==========================================================
        # ICEBERG
        # ==========================================================

        if writer_type == "iceberg":
            table = cfg.get(
                "table",
                "default.table",
            )

            table = WriterBuilder._resolve_storage_value(table)

            return writer_cls(
                table=table,
            )

        # ==========================================================
        # CONSOLE
        # ==========================================================

        if writer_type == "console":
            return writer_cls()

        raise ValueError(f"Unsupported writer type: {writer_type}")
