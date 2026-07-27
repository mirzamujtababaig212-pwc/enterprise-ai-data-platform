class StorageConfig:
    BRONZE_TABLE = "bronze"
    SILVER_TABLE = "silver"
    GOLD_TABLE = "gold"

    BRONZE_PATH = "/tmp/bronze"
    SILVER_PATH = "/tmp/silver"

    BRONZE_CHECKPOINT = "/tmp/checkpoints/bronze"
    SILVER_CHECKPOINT = "/tmp/checkpoints/silver"

    BRONZE_DLQ_TABLE = "bronze_dlq"
    SILVER_DLQ_TABLE = "silver_dlq"
