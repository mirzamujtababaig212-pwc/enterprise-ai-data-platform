from pathlib import Path

from common.config.storage import StorageConfig


def test_storage_config_defaults_to_local(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENTERPRISE_AI_PLATFORM_ROOT", raising=False)

    config = StorageConfig()

    assert config.BRONZE_PATH == "/app/data/delta/bronze/vehicle_events"
    assert config.SILVER_PATH == "/app/data/delta/silver/vehicle_events"
    assert config.GOLD_PATH == "/app/data/delta/gold/vehicle_metrics"


def test_storage_config_aws_uses_s3a_paths(monkeypatch):
    monkeypatch.setenv("APP_ENV", "AWS")

    config = StorageConfig()

    assert config.BRONZE_PATH == ("s3a://enterprise-data-ai-platform/bronze/vehicle_events")
    assert config.SILVER_PATH == ("s3a://enterprise-data-ai-platform/silver/vehicle_events")
    assert config.GOLD_PATH == ("s3a://enterprise-data-ai-platform/gold/vehicle_metrics")


def test_storage_config_aws_uses_s3a_checkpoints(monkeypatch):
    monkeypatch.setenv("APP_ENV", "AWS")

    config = StorageConfig()

    assert config.BRONZE_CHECKPOINT == (
        "s3a://enterprise-data-ai-platform/checkpoints/bronze_streaming"
    )
    assert config.SILVER_CHECKPOINT == (
        "s3a://enterprise-data-ai-platform/checkpoints/silver_streaming"
    )


def test_storage_config_aws_preserves_catalog_tables(monkeypatch):
    monkeypatch.setenv("APP_ENV", "AWS")

    config = StorageConfig()

    assert config.BRONZE_TABLE == "bronze.vehicle_events"
    assert config.SILVER_TABLE == "silver.vehicle_events"
    assert config.GOLD_TABLE == "gold.vehicle_metrics"


def test_storage_config_local_root_remains_supported(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "DEV")
    monkeypatch.setenv(
        "ENTERPRISE_AI_PLATFORM_ROOT",
        str(tmp_path),
    )

    config = StorageConfig()

    assert config.BRONZE_PATH == str(
        Path(tmp_path) / "data" / "delta" / "bronze" / "vehicle_events"
    )
    assert config.SILVER_PATH == str(
        Path(tmp_path) / "data" / "delta" / "silver" / "vehicle_events"
    )
    assert config.GOLD_PATH == str(Path(tmp_path) / "data" / "delta" / "gold" / "vehicle_metrics")
