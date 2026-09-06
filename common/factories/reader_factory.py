from common.builders.reader_builder import ReaderBuilder


class ReaderFactory:

    @staticmethod
    def create(config):
        return ReaderBuilder.build(config)
