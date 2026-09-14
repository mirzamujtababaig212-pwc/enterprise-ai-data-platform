from rag.loaders.databricks import DatabricksDocumentLoader
from rag.loaders.fabric import FabricDocumentLoader
from rag.loaders.gold import GoldVehicleMetricsDocumentLoader
from rag.loaders.s3 import S3DocumentLoader
from rag.loaders.snowflake import SnowflakeDocumentLoader
from rag.loaders.spark import SparkDataFrameDocumentLoader
from rag.loaders.vehicle_metrics import VehicleMetricsDocumentMapper

__all__ = [
    "DatabricksDocumentLoader",
    "FabricDocumentLoader",
    "S3DocumentLoader",
    "SnowflakeDocumentLoader",
    "SparkDataFrameDocumentLoader",
    "VehicleMetricsDocumentMapper",
    "GoldVehicleMetricsDocumentLoader",
]
