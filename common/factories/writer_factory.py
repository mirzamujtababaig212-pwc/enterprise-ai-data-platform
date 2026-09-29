from common.builders.writer_builder import WriterBuilder


class WriterFactory:
    @staticmethod
    def create(
        config,
        *,
        glue_synchronizer=None,
    ):
        return WriterBuilder.build(
            config,
            glue_synchronizer=glue_synchronizer,
        )
