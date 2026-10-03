import pytest

from common.config.config_loader import ConfigLoader


def test_load_dev():
    config = ConfigLoader.load("dev")

    assert config is not None
    assert config["environment"] == "dev"

    assert config["postgres"]["host"] == "localhost"
    assert config["postgres"]["port"] == 5432
    assert config["postgres"]["database"] == "vehicle_platform"

    assert config["kafka"]["bootstrap_servers"] == "localhost:9092"
    assert config["storage"]["bronze_table"] == "bronze.vehicle_events"


def test_load_qa():
    config = ConfigLoader.load("qa")
    assert config is not None


def test_invalid_environment():
    with pytest.raises(FileNotFoundError):
        ConfigLoader.load("invalid")


def test_config_loader_uses_configured_project_root(monkeypatch, tmp_path):
    config_root = tmp_path / "config" / "environments"
    config_root.mkdir(parents=True)

    (config_root / "aws.yaml").write_text(
        """
environment:
  name: aws
  cloud: aws

storage:
  raw: s3://example/raw
  bronze: s3://example/bronze
  silver: s3://example/silver
  gold: s3://example/gold
  checkpoints: s3://example/checkpoints
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("ENTERPRISE_AI_PLATFORM_ROOT", str(tmp_path))

    result = ConfigLoader.load("aws")

    assert result["environment"]["name"] == "aws"
    assert result["storage"]["bronze"] == "s3://example/bronze"
