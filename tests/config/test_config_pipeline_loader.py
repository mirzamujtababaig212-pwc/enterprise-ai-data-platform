from common.config.pipeline_loader import PipelineLoader


def test_pipeline_loader():
    result = PipelineLoader.load("bronze")

    assert result["pipeline"]["class"] == "bronze"
    assert result["reader"]["type"] == "kafka"
    assert result["writer"]["type"] == "delta"


def test_load_bronze_runtime_config():
    config = PipelineLoader.load_runtime_config("bronze")
    runtime = PipelineLoader.load("bronze")["runtime"]

    assert config.pipeline_name == "bronze"
    assert config.checkpoint == runtime["checkpoint"]
    assert config.output_mode == runtime["output_mode"]
    assert config.retries == runtime["retries"]
    assert config.retry_delay == runtime["retry_delay"]
    assert config.enable_validation == runtime["enable_validation"]
    assert config.enable_metrics == runtime["enable_metrics"]
    assert config.enable_dlq == runtime["enable_dlq"]


def test_load_silver_runtime_config():
    config = PipelineLoader.load_runtime_config("silver")
    runtime = PipelineLoader.load("silver")["runtime"]

    assert config.pipeline_name == "silver"
    assert config.checkpoint == runtime["checkpoint"]
    assert config.output_mode == runtime["output_mode"]
    assert config.retries == runtime["retries"]
    assert config.retry_delay == runtime["retry_delay"]
    assert config.enable_validation == runtime["enable_validation"]
    assert config.enable_metrics == runtime["enable_metrics"]
    assert config.enable_dlq == runtime["enable_dlq"]


def test_load_gold_runtime_config():
    config = PipelineLoader.load_runtime_config("gold")
    runtime = PipelineLoader.load("gold")["runtime"]

    assert config.pipeline_name == "gold"
    assert config.checkpoint == ""
    assert config.output_mode == runtime["output_mode"]
    assert config.retries == runtime["retries"]
    assert config.retry_delay == runtime["retry_delay"]
    assert config.enable_validation is False
    assert config.enable_metrics is True
    assert config.enable_dlq is False


def test_runtime_config_uses_defaults_for_unconfigured_fields():
    config = PipelineLoader.load_runtime_config("bronze")

    assert config.query_name is None
    assert config.trigger is None
