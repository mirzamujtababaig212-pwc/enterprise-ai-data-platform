from unittest.mock import Mock

from common.writers.console_writer import ConsoleWriter


def test_console_batch():
    writer = ConsoleWriter()
    df = Mock()
    writer.write_batch(df)
    df.show.assert_called_once_with(
        truncate=False
    )

def test_console_stream():
    writer = ConsoleWriter()
    df = Mock()
    stream = Mock()
    df.writeStream = stream
    stream.outputMode.return_value = stream
    stream.format.return_value = stream
    stream.option.return_value = stream
    stream.start.return_value = Mock()
    writer.write_stream(
        df,
        Mock()
    )
    stream.outputMode.assert_called_once_with(
        "append"
    )
    stream.format.assert_called_once_with(
        "console"
    )
    assert stream.option.call_count == 2
    stream.start.assert_called_once()
