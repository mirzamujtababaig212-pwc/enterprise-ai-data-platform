from common import constants


def test_storage_paths():
    assert constants.BRONZE_PATH == "data/bronze"
    assert constants.SILVER_PATH == "data/silver"
    assert constants.GOLD_PATH == "data/gold"


def test_checkpoints():
    assert constants.BRONZE_CHECKPOINT == "spark/checkpoints/bronze"
    assert constants.SILVER_CHECKPOINT == "spark/checkpoints/silver"
    assert constants.GOLD_CHECKPOINT == "spark/checkpoints/gold"


def test_kafka():
    assert constants.VEHICLE_TOPIC == "vehicle-telemetry"


def test_postgres():
    assert constants.POSTGRES_TABLE == "vehicle_gold"
