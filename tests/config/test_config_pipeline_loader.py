from common.config.pipeline_loader import PipelineLoader


def test_pipeline_loader():
    result = PipelineLoader.load("bronze")

    assert result["pipeline"]["class"] == "bronze"
    assert result["reader"]["type"] == "kafka"
    assert result["writer"]["type"] == "delta"


def test_load_bronze_runtime_config():
    config = PipelineLoader.load_runtime_config("bronze")

    assert config.pipeline_name == "bronze"
    assert config.checkpoint == PipelineLoader.load("bronze")["writer"]["checkpoint"]
    assert config.output_mode == "append"


def test_load_silver_runtime_config():
    config = PipelineLoader.load_runtime_config("silver")

    assert config.pipeline_name == "silver"
    assert config.checkpoint == PipelineLoader.load("silver")["writer"]["checkpoint"]
    assert config.output_mode == "append"


def test_load_gold_runtime_config():
    config = PipelineLoader.load_runtime_config("gold")

    assert config.pipeline_name == "gold"
    assert config.checkpoint == ""
    assert config.output_mode == "append"


def test_runtime_config_uses_defaults_for_unconfigured_fields():
    config = PipelineLoader.load_runtime_config("bronze")

    assert config.query_name is None
    assert config.trigger is None
    assert config.retries == 3
    assert config.retry_delay == 2
    assert config.enable_validation is True
    assert config.enable_metrics is True
    assert config.enable_dlq is True
