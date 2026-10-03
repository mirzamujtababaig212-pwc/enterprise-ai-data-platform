from unittest.mock import Mock, patch

from common.config.settings import Settings
from common.pipelines.pipeline_runtime_config import PipelineRuntimeConfig
from common.readers.csv_reader import CSVReader
from spark.batch.ingest.batch_to_bronze import BatchToBronzePipeline
from spark.schemas.bronze_schema import bronze_schema


def test_batch_to_bronze_uses_canonical_raw_csv_reader():
    spark = Mock()
    glue_synchronizer = Mock()

    with (
        patch("spark.batch.ingest.batch_to_bronze.CSVReader") as mock_reader,
        patch("spark.batch.ingest.batch_to_bronze.BatchBronzeTransformer") as mock_transformer,
        patch("spark.batch.ingest.batch_to_bronze.CompositeValidator") as mock_validator,
        patch("spark.batch.ingest.batch_to_bronze.DuplicateValidator"),
        patch("spark.batch.ingest.batch_to_bronze.BusinessRuleValidator"),
        patch("spark.batch.ingest.batch_to_bronze.build_aws_glue_synchronizer") as mock_glue,
        patch("spark.batch.ingest.batch_to_bronze.DeltaWriter") as mock_writer,
    ):
        mock_glue.return_value = glue_synchronizer

        pipeline = BatchToBronzePipeline(spark)

    mock_reader.assert_called_once_with(
        path=Settings.storage.RAW_VEHICLE_DATA_PATH,
        schema=bronze_schema,
    )
    mock_transformer.assert_called_once_with()
    mock_validator.assert_called_once()
    mock_glue.assert_called_once_with()

    mock_writer.assert_called_once_with(
        table=Settings.storage.BRONZE_TABLE,
        path=Settings.storage.BRONZE_PATH,
        mode="append",
        glue_synchronizer=glue_synchronizer,
        glue_database_name="enterprise_ai_platform",
        glue_table_name="vehicle_events",
    )

    assert pipeline.spark is spark
    assert pipeline.reader is mock_reader.return_value
    assert pipeline.transformer is mock_transformer.return_value
    assert pipeline.validator is mock_validator.return_value
    assert pipeline.writer is mock_writer.return_value
    assert isinstance(pipeline.config, PipelineRuntimeConfig)

    assert CSVReader is not None
