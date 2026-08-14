from unittest.mock import Mock, patch

import pytest

from tests.pipelines.dummy_pipeline import DummyPipeline


def make_pipeline(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    return DummyPipeline(
        spark=spark,
        reader=mock_reader,
        validator=mock_validator,
        writer=mock_writer,
        transformer=mock_transformer,
        metrics=mock_metrics,
        dlq=mock_dlq,
    )


def test_pipeline_creation(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    assert pipeline.spark is spark
    assert pipeline.reader is mock_reader
    assert pipeline.validator is mock_validator
    assert pipeline.writer is mock_writer
    assert pipeline.transformer is mock_transformer
    assert pipeline.metrics is mock_metrics
    assert pipeline.dlq is mock_dlq


def test_read(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    df = Mock()
    mock_reader.read.return_value = df

    result = pipeline.read()

    assert result is df

    mock_reader.read.assert_called_once_with(spark)


def test_validate(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    batch = Mock()
    valid = Mock()
    invalid = Mock()

    mock_validator.validate.return_value = (
        valid,
        invalid,
    )

    result_valid, result_invalid = pipeline.validate(batch)

    assert result_valid is valid
    assert result_invalid is invalid

    mock_validator.validate.assert_called_once_with(batch)


def test_handle_invalid_records(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    invalid = Mock()

    pipeline.handle_invalid_records(invalid)

    mock_dlq.write.assert_called_once_with(invalid)


def test_process_batch(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    batch = Mock()
    transformed = Mock()
    valid = Mock()
    invalid = Mock()

    mock_transformer.transform.return_value = transformed

    mock_validator.validate.return_value = (
        valid,
        invalid,
    )

    pipeline.process_batch(
        batch,
        10,
    )

    mock_transformer.transform.assert_called_once_with(batch)

    mock_validator.validate.assert_called_once_with(mock_transformer.transform.return_value)

    mock_writer.write.assert_called_once_with(valid)

    mock_dlq.write.assert_called_once_with(invalid)

    mock_metrics.record_batch.assert_called_once()


def test_process_batch_does_not_write_none(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    batch = Mock()
    transformed = Mock()

    mock_transformer.transform.return_value = transformed

    mock_validator.validate.return_value = (
        None,
        None,
    )

    pipeline.process_batch(
        batch,
        10,
    )

    mock_writer.write.assert_not_called()


def test_retry(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    batch = Mock()
    transformed = Mock()
    valid = Mock()
    invalid = Mock()

    mock_transformer.transform.return_value = transformed

    # Validation must return the two values expected by
    # BasePipeline.process_batch().
    mock_validator.validate.return_value = (
        valid,
        invalid,
    )

    # First write fails; second write succeeds.
    mock_writer.write.side_effect = [
        RuntimeError("temporary"),
        None,
    ]

    pipeline.process_batch(
        batch,
        1,
    )

    assert mock_transformer.transform.call_count == 2

    assert mock_validator.validate.call_count == 2

    assert mock_writer.write.call_count == 2

    mock_writer.write.assert_any_call(valid)

    assert mock_dlq.write.call_count == 1

    assert mock_metrics.record_batch.call_count == 1


def test_write_stream(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    df = Mock()
    query = Mock()

    mock_writer.write_stream.return_value = query
    query.awaitTermination.return_value = "terminated"

    result = pipeline.write_stream(df)

    mock_writer.write_stream.assert_called_once()

    assert result == "terminated"

    query.awaitTermination.assert_called_once()


def test_collect_metrics(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    df = Mock()

    pipeline.collect_metrics(
        "bronze",
        1,
        df,
        None,
    )

    mock_metrics.record_batch.assert_called_once()


def test_cleanup_on_batch_failure(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = make_pipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    mock_reader.read.side_effect = RuntimeError("read failed")

    with patch.object(
        pipeline,
        "cleanup",
    ) as mock_cleanup:

        with pytest.raises(
            RuntimeError,
            match="read failed",
        ):
            pipeline.run_batch()

        mock_cleanup.assert_called_once()
