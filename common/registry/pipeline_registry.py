from common.pipelines.bronze_pipeline import BronzePipeline
from common.pipelines.gold_pipeline import GoldPipeline
from common.pipelines.silver_pipeline import SilverPipeline

PIPELINE_REGISTRY = {
    "bronze": BronzePipeline,
    "silver": SilverPipeline,
    "gold": GoldPipeline,
}
