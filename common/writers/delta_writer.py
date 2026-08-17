from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from common.logging.logger import get_logger

logger = get_logger(__name__)


class DeltaWriter:
    """
    Canonical Delta Lake writer.

    Supports:

    1. Batch Delta writes
    2. Streaming Delta writes through foreachBatch
    3. Automatic Spark catalog registration
    4. Backward-compatible construction using only table=
    """

    VALID_MODES = {
        "append",
        "overwrite",
    }

    def __init__(
        self,
        table: str,
        path: str | None = None,
        mode: str = "append",
        checkpoint: str | None = None,
        output_mode: str | None = None,
    ) -> None:

        if not table or not table.strip():
            raise ValueError("Delta table name cannot be empty.")

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported Delta write mode: {mode}. " f"Valid modes: {sorted(self.VALID_MODES)}"
            )

        self.table = table.strip()

        self.path = Path(path).resolve() if path else self._resolve_default_path(self.table)

        self.mode = mode

        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None

        self.output_mode = output_mode

    # ==============================================================
    # DEFAULT PATH RESOLUTION
    # ==============================================================

    @staticmethod
    def _resolve_default_path(
        table: str,
    ) -> Path:
        """
        Resolve the canonical Delta path from StorageConfig.

        This keeps older callers that only provide table=
        compatible with the canonical storage configuration.
        """

        from common.config.storage import StorageConfig

        mapping = {
            StorageConfig.BRONZE_TABLE: StorageConfig.BRONZE_PATH,
            StorageConfig.SILVER_TABLE: StorageConfig.SILVER_PATH,
            StorageConfig.GOLD_TABLE: StorageConfig.GOLD_PATH,
        }

        if table not in mapping:
            raise ValueError(
                "No canonical Delta path configured for "
                f"table '{table}'. "
                "Provide path= explicitly."
            )

        return Path(mapping[table]).resolve()

    # ==============================================================
    # BATCH
    # ==============================================================

    def write(self, df):
        """
        Default write operation.
        """
        return self.write_batch(df)

    def write_batch(self, df):

        start = time.time()

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Writing Delta table=%s",
            self.table,
        )

        logger.info(
            "Delta path=%s",
            self.path,
        )

        logger.info(
            "Delta mode=%s",
            self.mode,
        )

        writer = df.write.format("delta").mode(self.mode)

        writer = writer.option(
            "overwriteSchema",
            "true",
        )

        # Prefer path-based writing because the platform
        # has canonical storage paths.
        writer.save(str(self.path))

        self._register_table(df.sparkSession)

        logger.info(
            "Delta batch write completed in %.2f sec",
            time.time() - start,
        )

    # ==============================================================
    # STREAMING
    # ==============================================================

    def write_stream(
        self,
        df,
        foreach_batch: Callable,
        checkpoint: str | None = None,
        output_mode: str | None = None,
        query_name: str | None = None,
        trigger: dict | None = None,
    ):
        """
        Start a streaming Delta write.

        Explicit checkpoint takes precedence over the
        checkpoint configured on the writer.
        """

        effective_checkpoint = checkpoint or (str(self.checkpoint) if self.checkpoint else None)

        if not effective_checkpoint:
            raise ValueError("Streaming Delta writes require " "a checkpoint path.")

        effective_output_mode = output_mode or self.output_mode or "append"

        checkpoint_path = Path(effective_checkpoint).resolve()

        checkpoint_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info("Starting streaming Delta write")

        logger.info(
            "Delta table=%s",
            self.table,
        )

        logger.info(
            "Delta path=%s",
            self.path,
        )

        logger.info(
            "Checkpoint=%s",
            checkpoint_path,
        )

        writer = (
            df.writeStream.outputMode(effective_output_mode)
            .option(
                "checkpointLocation",
                str(checkpoint_path),
            )
            .foreachBatch(foreach_batch)
        )

        if query_name:
            writer = writer.queryName(query_name)

        if trigger:

            if trigger.get("processingTime"):

                writer = writer.trigger(processingTime=trigger["processingTime"])

            elif trigger.get("availableNow"):

                writer = writer.trigger(availableNow=True)

            elif trigger.get("once"):

                writer = writer.trigger(once=True)

        return writer.start()

    # ==============================================================
    # CATALOG
    # ==============================================================

    def _register_table(
        self,
        spark,
    ):

        if spark.catalog.tableExists(self.table):

            logger.info(
                "Catalog table already exists: %s",
                self.table,
            )

            return

        database = self.table.split(
            ".",
            1,
        )[0]

        spark.sql(
            f"""
            CREATE DATABASE IF NOT EXISTS
            {database}
            """
        )

        logger.info(
            "Registering Delta table=%s at path=%s",
            self.table,
            self.path,
        )

        spark.sql(
            f"""
            CREATE TABLE {self.table}
            USING DELTA
            LOCATION '{self.path}'
            """
        )
