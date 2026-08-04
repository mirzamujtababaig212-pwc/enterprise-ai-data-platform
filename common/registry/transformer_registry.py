from common.transformers.base_transformer import BaseTransformer
from common.transformers.bronze_transformer import BronzeTransformer
from common.transformers.gold_transformer import GoldTransformer
from common.transformers.silver_transformer import SilverTransformer

TRANSFORMER_REGISTRY: dict[str, type[BaseTransformer]] = {
    "bronze": BronzeTransformer,
    "silver": SilverTransformer,
    "gold": GoldTransformer,
}
