from common.transformers.bronze_transformer import (
    BronzeTransformer,
)
from common.transformers.silver_transformer import (
    SilverTransformer,
)
from spark.transformations.silver_to_gold_transformer import (
    SilverToGoldTransformer,
)

TRANSFORMER_REGISTRY = {
    "bronze": BronzeTransformer,
    "silver": SilverTransformer,
    "gold": SilverToGoldTransformer,
}
