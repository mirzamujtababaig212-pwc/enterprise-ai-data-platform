from common.registry.transformer_registry import TRANSFORMER_REGISTRY


class TransformerFactory:
    @staticmethod
    def create(config):
        transformer_type = config["transformer"]["type"]
        if transformer_type not in TRANSFORMER_REGISTRY:
            raise ValueError(
                f"Unknown transformer type: {transformer_type}"
            )
        return TRANSFORMER_REGISTRY[transformer_type]()
