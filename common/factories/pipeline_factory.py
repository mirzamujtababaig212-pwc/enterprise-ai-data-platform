from common.factories.dlq_factory import DLQFactory
from common.factories.metrics_factory import MetricsFactory
from common.factories.reader_factory import ReaderFactory
from common.factories.transformer_factory import TransformerFactory
from common.factories.validator_factory import ValidatorFactory
from common.factories.writer_factory import WriterFactory
from common.loaders.pipeline_loader import PipelineLoader
from common.registry.pipeline_registry import PIPELINE_REGISTRY


class PipelineFactory:
    @staticmethod
    def get_pipeline(
        name,
        spark,
        *,
        glue_synchronizer=None,
        run_id=None,
    ):
        config = PipelineLoader.load(name)
        pipeline_name = config["pipeline"]["class"]
        if pipeline_name not in PIPELINE_REGISTRY:
            raise ValueError(f"Unknown pipeline class '{pipeline_name}'")
        pipeline_cls = PIPELINE_REGISTRY[pipeline_name]
        return pipeline_cls(
            spark=spark,
            reader=ReaderFactory.create(config),
            writer=WriterFactory.create(
                config,
                glue_synchronizer=glue_synchronizer,
            ),
            transformer=TransformerFactory.create(config),
            validator=ValidatorFactory.create(config),
            metrics=MetricsFactory.create(config),
            dlq=DLQFactory.create(config),
            config=PipelineLoader.load_runtime_config(name),
            run_id=run_id,
        )
