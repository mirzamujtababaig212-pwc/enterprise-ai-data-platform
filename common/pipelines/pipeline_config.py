from dataclasses import dataclass

from pyspark.sql.types import StructType


@dataclass
class PipelineConfig:
    pipeline_name: str
    source: str
    path: str | None = None
    trigger: dict | None = None
    watermark: str | None = None
    query_name: str | None = None
    partition_columns: list[str] | None = None
    target: str = "delta"
    table: str = ""
    checkpoint: str = ""
    schema: StructType | None= None
    output_mode: str = "append"
    retries: int = 3
    retry_delay: int = 2
    enable_validation: bool = True
    enable_metrics: bool = True
    enable_dlq: bool = True
    description: str = ""
    version: str = "1.0"
