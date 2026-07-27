from common.registry.reader_registry import READER_REGISTRY


class ReaderFactory:
    @staticmethod
    def create(config):
        reader_cfg = config["reader"]
        reader_type = reader_cfg["type"]
        if reader_type not in READER_REGISTRY:
            raise ValueError(f"Unknown reader type: {reader_type}")
        reader_cls = READER_REGISTRY[reader_type]
        kwargs = {
            k: v
            for k, v in reader_cfg.items()
            if k != "type"
        }
        return reader_cls(**kwargs)
