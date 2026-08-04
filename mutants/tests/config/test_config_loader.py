import pytest

from common.config.config_loader import ConfigLoader


def test_load_dev():
    config = ConfigLoader.load("dev")

    assert config is not None
    assert config["environment"] == "dev"

    assert config["postgres"]["host"] == "localhost"
    assert config["postgres"]["port"] == 5432
    assert config["postgres"]["database"] == "enterprise"

    assert config["kafka"]["bootstrap_servers"] == "localhost:9092"
    assert config["storage"]["bronze_table"] == "bronze_vehicle"


def test_load_qa():
    config = ConfigLoader.load("qa")
    assert config is not None


def test_invalid_environment():
    with pytest.raises(FileNotFoundError):
        ConfigLoader.load("invalid")
