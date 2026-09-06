from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineRuntimeConfig:
    pipeline_name: str
    checkpoint: str = ""
    output_mode: str = "append"
    query_name: str | None = None
    trigger: dict | None = None
    retries: int = 3
    retry_delay: int = 2
    enable_validation: bool = True
    enable_metrics: bool = True
    enable_dlq: bool = True
