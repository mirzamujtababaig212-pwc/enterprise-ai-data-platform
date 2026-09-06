from common.builders.writer_builder import WriterBuilder


class WriterFactory:
    @staticmethod
    def create(config):
        return WriterBuilder.build(config)
