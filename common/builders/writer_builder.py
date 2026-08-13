from common.config.settings import Settings


class WriterBuilder:

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
    def build(writer_cls, config):

        if not config:
            raise ValueError("Writer configuration cannot be empty.")

        cfg = config.get(
            "writer",
            config,
        )

        writer_type = cfg.get("type")

        if not writer_type:
            raise ValueError("Writer type is required.")

        writer_type = writer_type.lower()

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
                checkpoint=checkpoint,
                mode=mode,
                output_mode=output_mode,
            )

        # ==========================================================
        # POSTGRES
        # ==========================================================

        if writer_type == "postgres":

            return writer_cls(
                url=Settings.postgres.URL,
                table=Settings.postgres.TABLE,
                properties=Settings.postgres.PROPERTIES,
            )

        # ==========================================================
        # S3
        # ==========================================================

        if writer_type == "s3":

            path = WriterBuilder._resolve_storage_value(
                cfg.get("path"),
                Settings.storage.BRONZE_PATH,
            )

            return writer_cls(path=path)

        # ==========================================================
        # ICEBERG
        # ==========================================================

        if writer_type == "iceberg":

            table = cfg.get(
                "table",
                "default.table",
            )

            table = WriterBuilder._resolve_storage_value(table)

            return writer_cls(table=table)

        # ==========================================================
        # CONSOLE
        # ==========================================================

        if writer_type == "console":

            return writer_cls()

        raise ValueError(f"Unsupported writer type: {writer_type}")
