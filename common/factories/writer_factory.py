from common.registry.writer_registry import WRITER_REGISTRY


class WriterFactory:
    @staticmethod
    def create(config):
        writer_cfg = config["writer"]
        writer_type = writer_cfg["type"]
        if writer_type not in WRITER_REGISTRY:
            raise ValueError(f"Unknown writer type: {writer_type}")
        writer_cls = WRITER_REGISTRY[writer_type]
        kwargs = {k: v for k, v in writer_cfg.items() if k != "type"}
        return writer_cls(**kwargs)
