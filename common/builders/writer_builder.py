from common.config.settings import Settings


class WriterBuilder:
    @staticmethod
    def build(
        writer_cls,
        config
    ):
        cfg = config["writer"]
        writer_type = cfg["type"]
        if writer_cls is None:
           raise ValueError(
              f"Unknown writer type '{writer_type}'"
           )
        kwargs = {}
        if "table" in cfg:
            table = cfg["table"]
            if hasattr(
                Settings.storage,
                table
            ):
                kwargs["table"] = getattr(
                    Settings.storage,
                    table
                )
            else:
                kwargs["table"] = table
        if "checkpoint" in cfg:
            kwargs["checkpoint"] = getattr(
                Settings.storage,
                cfg["checkpoint"]
            )
        if "mode" in cfg:
            kwargs["mode"] = cfg["mode"]
        if "output_mode" in cfg:
            kwargs["output_mode"] = cfg["output_mode"]
        if "path" in cfg:
            kwargs["path"] = getattr(
                 Settings.storage,
                 cfg["path"],
                 cfg["path"]
            )
        if writer_type == "postgres":
            kwargs["url"] = Settings.postgres.URL
            kwargs["table"] = Settings.postgres.TABLE
            kwargs["properties"] = Settings.postgres.PROPERTIES
        elif writer_type == "delta":
            kwargs["table"] = getattr(
                Settings.storage,
                cfg.get("table", "BRONZE_TABLE")
            )
            kwargs["checkpoint"] = getattr(
                Settings.storage,
                cfg.get(
                    "checkpoint",
                    "BRONZE_CHECKPOINT"
                )
            )
            kwargs["mode"] = cfg.get(
                "mode",
                "append"
            )
            kwargs["output_mode"] = cfg.get(
                "output_mode",
                "append"
            )
        elif writer_type == "s3":
            kwargs["path"] = getattr(
            Settings.storage,
            cfg.get("path", "BRONZE_PATH")
            )
        elif writer_type == "iceberg":
            kwargs["table"] = cfg.get(
               "table",
               "default.table"
            )
        return writer_cls(**kwargs)
