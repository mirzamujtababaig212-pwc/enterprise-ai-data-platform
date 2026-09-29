from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from delta.tables import DeltaTable

from common.aws.catalog_synchronizer import AwsGlueCatalogSynchronizer
from common.aws.glue_table_input import build_delta_table_input
from common.logging.logger import get_logger
from common.writers.base_writer import BaseWriter

logger = get_logger(__name__)


class DeltaWriter(BaseWriter):
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
        "merge",
        "overwrite",
    }

    def __init__(
        self,
        table: str,
        path: str | None = None,
        mode: str = "append",
        checkpoint: str | None = None,
        output_mode: str | None = None,
        merge_keys: list[str] | None = None,
        glue_synchronizer: AwsGlueCatalogSynchronizer | None = None,
        glue_database_name: str | None = None,
        glue_table_name: str | None = None,
    ) -> None:

        if not table or not table.strip():
            raise ValueError("Delta table name cannot be empty.")

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported Delta write mode: {mode}. " f"Valid modes: {sorted(self.VALID_MODES)}"
            )

        self.table = table.strip()

        self.path = self._normalize_path(path) if path else self._resolve_default_path(self.table)

        self.mode = mode

        if merge_keys is not None:
            merge_keys = [str(key).strip() for key in merge_keys if str(key).strip()]

        if mode == "merge" and not merge_keys:
            raise ValueError("Delta merge mode requires at least one merge key.")

        if mode != "merge" and merge_keys:
            raise ValueError("Delta merge keys are only valid with merge mode.")

        if merge_keys and len(merge_keys) != len(set(merge_keys)):
            raise ValueError("Delta merge keys must be unique.")

        self.merge_keys = merge_keys or []

        self.checkpoint = self._normalize_path(checkpoint) if checkpoint else None

        self.output_mode = output_mode

        self._glue_synchronizer = glue_synchronizer
        self._glue_database_name = (
            glue_database_name.strip()
            if glue_database_name and glue_database_name.strip()
            else None
        )
        self._glue_table_name = (
            glue_table_name.strip() if glue_table_name and glue_table_name.strip() else None
        )

        if self._glue_synchronizer is not None and not self._glue_database_name:
            raise ValueError("Glue database name is required when Glue synchronization is enabled.")

    # ==============================================================
    # PATH NORMALIZATION
    # ==============================================================

    @staticmethod
    def _is_cloud_uri(value: str) -> bool:
        value = str(value)
        return "://" in value or value.startswith("dbfs:/")

    @classmethod
    def _normalize_path(cls, value: str):
        value = str(value)

        if cls._is_cloud_uri(value):
            return value

        return Path(value).resolve()

    @staticmethod
    def _ensure_local_parent(path) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

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

        from common.config.settings import Settings

        mapping = {
            Settings.storage.BRONZE_TABLE: Settings.storage.BRONZE_PATH,
            Settings.storage.SILVER_TABLE: Settings.storage.SILVER_PATH,
            Settings.storage.GOLD_TABLE: Settings.storage.GOLD_PATH,
        }

        if table not in mapping:
            raise ValueError(
                "No canonical Delta path configured for "
                f"table '{table}'. "
                "Provide path= explicitly."
            )

        return DeltaWriter._normalize_path(mapping[table])

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

        self._ensure_local_parent(self.path)

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

        if self.mode == "merge":
            self._merge_batch(df)
        else:
            writer = df.write.format("delta").mode(self.mode)

            writer = writer.option(
                "overwriteSchema",
                "true",
            )

            # Prefer path-based writing because the platform
            # has canonical storage paths.
            writer.save(str(self.path))

            self._register_table(df.sparkSession)

        self._sync_glue_catalog(df)

        logger.info(
            "Delta batch write completed in %.2f sec",
            time.time() - start,
        )

    def _merge_batch(self, df):
        spark = df.sparkSession
        path = str(self.path)

        if not DeltaTable.isDeltaTable(spark, path):
            logger.info(
                "Delta merge target does not exist; " "initializing table=%s",
                self.table,
            )

            (df.write.format("delta").mode("append").option("overwriteSchema", "true").save(path))

            self._register_table(spark)
            self._sync_glue_catalog(df)
            return

        merge_condition = " AND ".join(
            f"target.`{key}` = source.`{key}`" for key in self.merge_keys
        )

        logger.info(
            "Executing Delta MERGE table=%s keys=%s",
            self.table,
            self.merge_keys,
        )

        target = DeltaTable.forPath(
            spark,
            path,
        )

        (
            target.alias("target")
            .merge(
                df.alias("source"),
                merge_condition,
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

        self._register_table(spark)
        self._sync_glue_catalog(df)

    # ==============================================================
    # AWS GLUE CATALOG
    # ==============================================================

    def _sync_glue_catalog(self, df) -> None:
        """
        Synchronize the successful Delta write with AWS Glue.

        Glue synchronization is deliberately optional so local
        environments and existing callers remain AWS-independent.
        """

        if self._glue_synchronizer is None:
            return

        if not self._glue_database_name:
            raise RuntimeError("Glue database name is required for Glue synchronization.")

        glue_table_name = self._glue_table_name or self.table.split(".", 1)[-1]

        table_input = build_delta_table_input(
            table_name=glue_table_name,
            schema=df.schema,
            location=str(self.path),
            description=f"Delta table for {self.table}.",
        )

        result = self._glue_synchronizer.sync(
            database_name=self._glue_database_name,
            table_input=table_input,
        )

        logger.info(
            "AWS Glue catalog synchronized: database=%s table=%s result=%s",
            self._glue_database_name,
            glue_table_name,
            result,
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

        if self.mode == "merge":
            raise ValueError("Delta merge mode is only supported for batch writes.")

        effective_checkpoint = checkpoint or (str(self.checkpoint) if self.checkpoint else None)

        if not effective_checkpoint:
            raise ValueError("Streaming Delta writes require " "a checkpoint path.")

        effective_output_mode = output_mode or self.output_mode or "append"

        checkpoint_path = self._normalize_path(effective_checkpoint)

        self._ensure_local_parent(checkpoint_path)
        self._ensure_local_parent(self.path)

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

    @staticmethod
    def _extract_catalog_location(rows) -> str | None:
        for row in rows:
            col_name = getattr(row, "col_name", None)

            if col_name == "Location":
                return getattr(row, "data_type", None)

        return None

    @staticmethod
    def _locations_match(expected, actual) -> bool:
        expected = str(expected).rstrip("/")
        actual = str(actual).rstrip("/")

        if expected == actual:
            return True

        if expected.startswith("s3://") and actual.startswith("s3a://"):
            return expected[5:] == actual[6:]

        if expected.startswith("s3a://") and actual.startswith("s3://"):
            return expected[6:] == actual[5:]

        expected_local = (
            Path(expected[5:]).resolve()
            if expected.startswith("file:")
            else Path(expected).resolve() if "://" not in expected else None
        )

        actual_local = Path(actual[5:]).resolve() if actual.startswith("file:") else None

        if expected_local is not None and actual_local is not None:
            return expected_local == actual_local

        return False

    def _register_table(
        self,
        spark,
    ):
        if spark.catalog.tableExists(self.table):
            rows = spark.sql(f"DESCRIBE EXTENDED {self.table}").collect()

            catalog_location = self._extract_catalog_location(rows)

            if not catalog_location:
                raise RuntimeError(
                    "Delta catalog location could not be determined " f"for table '{self.table}'."
                )

            if not self._locations_match(
                self.path,
                catalog_location,
            ):
                raise RuntimeError(
                    "Delta catalog location mismatch for "
                    f"table '{self.table}': "
                    f"expected '{self.path}', "
                    f"catalog points to '{catalog_location}'."
                )

            logger.info(
                "Catalog table already exists at expected location: %s",
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
