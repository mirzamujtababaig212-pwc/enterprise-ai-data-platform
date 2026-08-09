class TransformerBuilder:
    @staticmethod
    def build(transformer_cls, config):

        if transformer_cls is None:
            raise ValueError("Invalid Transformer")

        return transformer_cls()
