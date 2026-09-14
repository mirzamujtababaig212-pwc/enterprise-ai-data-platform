from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from common.config.config_loader import ConfigLoader


class StorageConfig:
    """
    Centralized storage configuration for the Enterprise AI Platform.

    Storage paths are resolved from the active application environment.

    Local environments retain the existing filesystem layout while cloud
    environments can resolve their canonical storage paths from
    config/environments/<environment>.yaml.
    """

    BRONZE_DATABASE = "bronze"
    SILVER_DATABASE = "silver"
    GOLD_DATABASE = "gold"

    BRONZE_TABLE = "bronze.vehicle_events"
    SILVER_TABLE = "silver.vehicle_events"
    GOLD_TABLE = "gold.vehicle_metrics"

    BRONZE_DLQ_TABLE = "bronze.bronze_dlq"
    SILVER_DLQ_TABLE = "silver.silver_dlq"

    POSTGRES_TABLE = "vehicle_metrics"

    def __init__(
        self,
        environment: str | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.environment = (environment or os.getenv("APP_ENV", "DEV")).strip().lower()

        self.project_root = Path(
            project_root
            if project_root is not None
            else os.getenv(
                "ENTERPRISE_AI_PLATFORM_ROOT",
                "/app",
            )
        )

        if self.environment == "aws":
            self._configure_aws()
        else:
            self._configure_local()

    # ==============================================================
    # LOCAL STORAGE
    # ==============================================================

    def _configure_local(self) -> None:
        root = self.project_root

        self.BATCH_INPUT_PATH = str(root / "data" / "bronze")
        self.BATCH_BRONZE_PATH = self.BATCH_INPUT_PATH

        self.RAW_VEHICLE_DATA_PATH = str(root / "data" / "vehicle_events.csv")

        self.BRONZE_PATH = str(root / "data" / "delta" / "bronze" / "vehicle_events")

        self.SILVER_PATH = str(root / "data" / "delta" / "silver" / "vehicle_events")

        self.GOLD_PATH = str(root / "data" / "delta" / "gold" / "vehicle_metrics")

        self.BRONZE_CHECKPOINT = str(root / "spark" / "checkpoints" / "bronze_streaming")

        self.SILVER_CHECKPOINT = str(root / "spark" / "checkpoints" / "silver_streaming")

        self.BATCH_BRONZE_CHECKPOINT = str(root / "spark" / "checkpoints" / "bronze_batch")

        self.BATCH_SILVER_CHECKPOINT = str(root / "spark" / "checkpoints" / "silver_batch")

    # ==============================================================
    # AWS STORAGE
    # ==============================================================

    def _configure_aws(self) -> None:
        config = ConfigLoader.load("aws")

        storage = config.get("storage")

        if not isinstance(storage, dict):
            raise ValueError("AWS environment configuration requires a storage mapping.")

        required = (
            "raw",
            "bronze",
            "silver",
            "gold",
            "checkpoints",
        )

        missing = [key for key in required if not storage.get(key)]

        if missing:
            raise ValueError("AWS storage configuration is missing: " + ", ".join(missing))

        raw = self._to_s3a_uri(storage["raw"])
        bronze = self._to_s3a_uri(storage["bronze"])
        silver = self._to_s3a_uri(storage["silver"])
        gold = self._to_s3a_uri(storage["gold"])
        checkpoints = self._to_s3a_uri(storage["checkpoints"])

        self.BATCH_INPUT_PATH = raw
        self.BATCH_BRONZE_PATH = bronze

        self.RAW_VEHICLE_DATA_PATH = f"{raw}/vehicle_events.csv"

        self.BRONZE_PATH = f"{bronze}/vehicle_events"
        self.SILVER_PATH = f"{silver}/vehicle_events"
        self.GOLD_PATH = f"{gold}/vehicle_metrics"

        self.BRONZE_CHECKPOINT = f"{checkpoints}/bronze_streaming"

        self.SILVER_CHECKPOINT = f"{checkpoints}/silver_streaming"

        self.BATCH_BRONZE_CHECKPOINT = f"{checkpoints}/bronze_batch"

        self.BATCH_SILVER_CHECKPOINT = f"{checkpoints}/silver_batch"

    # ==============================================================
    # URI NORMALIZATION
    # ==============================================================

    @staticmethod
    def _to_s3a_uri(value: Any) -> str:
        uri = str(value).rstrip("/")

        if uri.startswith("s3a://"):
            return uri

        if uri.startswith("s3://"):
            return "s3a://" + uri[len("s3://") :]

        raise ValueError("AWS storage paths must use s3:// or s3a:// URIs: " f"{value}")
