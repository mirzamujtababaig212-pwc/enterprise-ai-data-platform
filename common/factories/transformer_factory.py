from typing import cast

from common.registry.transformer_registry import TRANSFORMER_REGISTRY
from common.transformers.base_transformer import BaseTransformer


class TransformerFactory:
    @staticmethod
    def create(config) -> BaseTransformer:
        transformer_cfg = config["transformer"]
        transformer_type = transformer_cfg["type"]
        if transformer_type not in TRANSFORMER_REGISTRY:
            raise ValueError(f"Unknown transformer type: {transformer_type}")
        transformer_cls = cast(
            type[BaseTransformer],
            TRANSFORMER_REGISTRY[transformer_type],
        )
        kwargs = {k: v for k, v in transformer_cfg.items() if k != "type"}
        return transformer_cls(**kwargs)
