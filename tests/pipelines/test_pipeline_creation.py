from unittest.mock import Mock

from tests.pipelines.dummy_pipeline import DummyPipeline


def test_pipeline_creation(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq
):
    pipeline = DummyPipeline(
        spark=spark,
        reader=mock_reader,
        validator=mock_validator,
        writer=mock_writer,
        transformer=mock_transformer,
        metrics=mock_metrics,
        dlq=mock_dlq
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
        mock_dlq
):
    pipeline = DummyPipeline(
        spark=spark,
        reader=mock_reader,
        validator=mock_validator,
        writer=mock_writer,
        transformer=mock_transformer,
        metrics=mock_metrics,
        dlq=mock_dlq
    )
    df = Mock()
    transformed = Mock()
    mock_reader.read.return_value = df
    mock_transformer.transform.return_value = transformed
    pipeline.write_stream = Mock()
    pipeline.run()
    mock_reader.read.assert_called_once()
    mock_transformer.transform.assert_called_once_with(df)
    pipeline.write_stream.assert_called_once_with(transformed)

def test_validate(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq
    )
    valid = Mock()
    invalid = Mock()
    mock_validator.validate.return_value = (
        valid,
        invalid
    )
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
        mock_dlq
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq
    )
    df = Mock()
    pipeline.collect_metrics(
        "bronze",
        1,
        df,
        None
    )
    mock_metrics.record_batch.assert_called_once()

def test_handle_invalid_records(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq
    )
    invalid = Mock()
    pipeline.handle_invalid_records(invalid)
    mock_dlq.write.assert_called_once_with(invalid)

def test_write(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq
    )
    df = Mock()
    pipeline.write(df)
    mock_writer.write_batch.assert_called_once_with(df)

def test_process_batch(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq
    )
    batch = Mock()
    valid = Mock()
    invalid = Mock()
    output = Mock()
    mock_validator.validate.return_value = (
        valid,
        invalid
    )
    mock_transformer.transform.return_value = output
    pipeline.process_batch(
        batch,
        10
    )
    assert mock_validator.validate.call_count == 1
    mock_validator.validate.assert_called_with(mock_transformer.transform.return_value)
    assert mock_transformer.transform.call_count == 2
    mock_transformer.transform.assert_any_call(batch)
    mock_transformer.transform.assert_any_call(valid)
    mock_writer.write.assert_called_once_with(output)
    mock_dlq.write.assert_called_once_with(invalid)
    mock_metrics.record_batch.assert_called_once()

def test_retry(
        spark,
        mock_reader,
        mock_writer,
        mock_validator,
        mock_transformer,
        mock_metrics,
        mock_dlq
):
    pipeline = DummyPipeline(
        spark,
        mock_reader,
        mock_validator,
        mock_writer,
        mock_transformer,
        mock_metrics,
        mock_dlq
    )
    batch = Mock()
    valid = Mock()
    invalid = Mock()
    output = Mock()
    mock_validator.validate.return_value = (
        valid,
        invalid
    )
    mock_transformer.transform.return_value = output
    mock_writer.write.side_effect = [
        Exception("temporary"),
        None
    ]
    pipeline.process_batch(
        batch,
        1
    )
    assert mock_writer.write.call_count == 2

