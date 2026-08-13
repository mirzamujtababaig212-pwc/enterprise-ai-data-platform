from pathlib import Path


class StorageConfig:
    """
    Centralized storage configuration for the Enterprise AI Platform.

    Canonical Medallion Architecture
    ---------------------------------

        Historical Parquet
                |
                v
        Bronze Delta
                |
                v
        Silver Delta
                |
                v
        Gold Delta
                |
                v
        PostgreSQL serving layer

    All Delta filesystem paths and Spark catalog table names
    are defined here so that pipelines, readers, writers and
    configuration files do not independently redefine them.
    """

    # ==============================================================
    # PROJECT ROOT
    # ==============================================================

    PROJECT_ROOT = Path("/home/annie/enterprise_ai_platform")

    # ==============================================================
    # SOURCE DATA
    # ==============================================================

    # Historical batch input.
    #
    # Existing source files:
    #
    #   data/bronze/*.parquet
    #
    BATCH_INPUT_PATH = str(PROJECT_ROOT / "data" / "bronze")

    # Backward-compatible alias.
    BATCH_BRONZE_PATH = BATCH_INPUT_PATH

    # ==============================================================
    # DELTA STORAGE
    # ==============================================================

    BRONZE_PATH = str(PROJECT_ROOT / "data" / "delta" / "bronze" / "vehicle_events")

    SILVER_PATH = str(PROJECT_ROOT / "data" / "delta" / "silver" / "vehicle_events")

    GOLD_PATH = str(PROJECT_ROOT / "data" / "delta" / "gold" / "vehicle_metrics")

    # ==============================================================
    # SPARK CATALOG DATABASES
    # ==============================================================

    BRONZE_DATABASE = "bronze"
    SILVER_DATABASE = "silver"
    GOLD_DATABASE = "gold"

    # ==============================================================
    # SPARK CATALOG TABLES
    # ==============================================================

    BRONZE_TABLE = "bronze.vehicle_events"
    SILVER_TABLE = "silver.vehicle_events"
    GOLD_TABLE = "gold.vehicle_metrics"

    # ==============================================================
    # STREAMING CHECKPOINTS
    # ==============================================================

    BRONZE_CHECKPOINT = str(PROJECT_ROOT / "spark" / "checkpoints" / "bronze_streaming")

    SILVER_CHECKPOINT = str(PROJECT_ROOT / "spark" / "checkpoints" / "silver_streaming")

    # ==============================================================
    # BATCH CHECKPOINTS
    # ==============================================================

    BATCH_BRONZE_CHECKPOINT = str(PROJECT_ROOT / "spark" / "checkpoints" / "bronze_batch")

    BATCH_SILVER_CHECKPOINT = str(PROJECT_ROOT / "spark" / "checkpoints" / "silver_batch")

    # ==============================================================
    # DEAD LETTER QUEUES
    # ==============================================================

    BRONZE_DLQ_TABLE = "bronze.bronze_dlq"
    SILVER_DLQ_TABLE = "silver.silver_dlq"

    # ==============================================================
    # POSTGRESQL SERVING LAYER
    # ==============================================================

    POSTGRES_TABLE = "vehicle_metrics"
