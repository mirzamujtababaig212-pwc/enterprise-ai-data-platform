from common.builders.writer_builder import WriterBuilder
from common.registry.writer_registry import WRITER_REGISTRY


class WriterFactory:
    @staticmethod
    def create(config):
        writer_cfg = config["writer"]

        writer_type = str(writer_cfg["type"]).lower().strip()

        if writer_type not in WRITER_REGISTRY:
            raise ValueError(f"Unknown writer type: {writer_type}")

        writer_cls = WRITER_REGISTRY[writer_type]

        return WriterBuilder.build(
            writer_cls,
            config,
        )
