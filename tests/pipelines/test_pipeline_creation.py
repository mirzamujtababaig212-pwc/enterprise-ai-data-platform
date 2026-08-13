from unittest.mock import Mock, patch

from tests.pipelines.dummy_pipeline import DummyPipeline


def test_pipeline_creation(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = DummyPipeline(
        spark=spark,
        reader=mock_reader,
        validator=mock_validator,
        writer=mock_writer,
        transformer=mock_transformer,
        metrics=mock_metrics,
        dlq=mock_dlq,
    )

    assert pipeline.spark is spark
    assert pipeline.reader is mock_reader
    assert pipeline.validator is mock_validator
    assert pipeline.writer is mock_writer
    assert pipeline.transformer is mock_transformer
    assert pipeline.metrics is mock_metrics
    assert pipeline.dlq is mock_dlq


def test_run(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = DummyPipeline(
        spark=spark,
        reader=mock_reader,
        validator=mock_validator,
        writer=mock_writer,
        transformer=mock_transformer,
        metrics=mock_metrics,
        dlq=mock_dlq,
    )
    df = Mock()
    mock_reader.read.return_value = df
    with patch.object(
        pipeline,
        "process_batch",
    ) as mock_process_batch:
        pipeline.run(mode="batch")
    mock_reader.read.assert_called_once_with(spark)
    mock_process_batch.assert_called_once()
    call = mock_process_batch.call_args
    assert call.kwargs["batch_df"] is df
    assert "batch_id" in call.kwargs


def test_validate(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )
    valid = Mock()
    invalid = Mock()
    mock_validator.validate.return_value = (valid, invalid)
    v, i = pipeline.validate(Mock())
    assert v is valid
    assert i is invalid


def test_collect_metrics(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )
    df = Mock()
    pipeline.collect_metrics("bronze", 1, df, None)
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
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )
    invalid = Mock()
    pipeline.handle_invalid_records(invalid)
    mock_dlq.write.assert_called_once_with(invalid)


def test_write_valid_records(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )

    df = Mock()

    mock_transformer.transform.return_value = df
    mock_validator.validate.return_value = (df, None)

    pipeline.process_batch(df, 1)

    mock_writer.write.assert_called_once_with(df)


def test_process_batch(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )
    batch = Mock()
    valid = Mock()
    invalid = Mock()
    output = Mock()
    mock_validator.validate.return_value = (valid, invalid)
    mock_transformer.transform.return_value = output
    pipeline.process_batch(batch, 10)
    mock_transformer.transform.assert_called_once_with(batch)
    mock_validator.validate.assert_called_once_with(output)
    mock_writer.write.assert_called_once_with(valid)
    mock_dlq.write.assert_called_once_with(invalid)
    mock_metrics.record_batch.assert_called_once()


def test_retry(
    spark,
    mock_reader,
    mock_writer,
    mock_validator,
    mock_transformer,
    mock_metrics,
    mock_dlq,
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq,
    )
    batch = Mock()
    valid = Mock()
    invalid = Mock()
    output = Mock()
    mock_validator.validate.return_value = (valid, invalid)
    mock_transformer.transform.return_value = output
    mock_writer.write.side_effect = [RuntimeError("temporary"), None]
    pipeline.process_batch(batch, 1)
    assert mock_writer.write.call_count == 2
