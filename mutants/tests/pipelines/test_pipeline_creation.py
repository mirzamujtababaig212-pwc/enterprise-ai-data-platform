from unittest.mock import Mock, patch

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
    assert pipeline.config is DummyPipeline.CONFIG


def test_run_stream(
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

    mock_reader.read.return_value = df
    mock_writer.write_stream.return_value = query

    with patch.object(
        pipeline,
        "write_stream",
        return_value=query,
    ) as mock_write_stream:
        pipeline.run(mode="stream")

    mock_reader.read.assert_called_once_with(spark)
    mock_write_stream.assert_called_once_with(df)
    mock_transformer.transform.assert_not_called()


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

    valid = Mock()
    invalid = Mock()

    mock_validator.validate.return_value = (
        valid,
        invalid,
    )

    result_valid, result_invalid = pipeline.validate(Mock())

    assert result_valid is valid
    assert result_invalid is invalid

    mock_validator.validate.assert_called_once()


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
        mock_writer,
        mock_validator,
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
    mock_validator.validate.assert_called_once_with(transformed)
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
    valid = Mock()
    invalid = Mock()
    transformed = Mock()

    mock_validator.validate.return_value = (
        valid,
        invalid,
    )

    mock_transformer.transform.return_value = transformed

    mock_writer.write.side_effect = [
        RuntimeError("temporary"),
        None,
    ]

    pipeline.process_batch(
        batch,
        1,
    )

    assert mock_writer.write.call_count == 2
