from common.writers.streaming import apply_stream_options


class FakeStreamWriter:
    def __init__(self):
        self.calls = []

    def foreachBatch(self, value):
        self.calls.append(("foreachBatch", value))
        return self

    def option(self, key, value):
        self.calls.append(("option", key, value))
        return self

    def outputMode(self, value):
        self.calls.append(("outputMode", value))
        return self

    def queryName(self, value):
        self.calls.append(("queryName", value))
        return self

    def trigger(self, **kwargs):
        self.calls.append(("trigger", kwargs))
        return self

    def start(self):
        self.calls.append(("start",))
        return "query"


def test_apply_stream_options():
    writer = FakeStreamWriter()

    result = apply_stream_options(
        writer=writer,
        checkpoint="/tmp/checkpoint",
        output_mode="append",
        query_name="vehicle-stream",
        trigger={
            "processingTime": "10 seconds",
        },
    )

    assert result is writer

    assert (
        "option",
        "checkpointLocation",
        "/tmp/checkpoint",
    ) in writer.calls

    assert (
        "outputMode",
        "append",
    ) in writer.calls

    assert (
        "queryName",
        "vehicle-stream",
    ) in writer.calls

    assert (
        "trigger",
        {"processingTime": "10 seconds"},
    ) in writer.calls


def test_apply_stream_options_available_now():
    writer = FakeStreamWriter()

    apply_stream_options(
        writer=writer,
        trigger={
            "availableNow": True,
        },
    )

    assert (
        "trigger",
        {"availableNow": True},
    ) in writer.calls


def test_apply_stream_options_once():
    writer = FakeStreamWriter()

    apply_stream_options(
        writer=writer,
        trigger={
            "once": True,
        },
    )

    assert (
        "trigger",
        {"once": True},
    ) in writer.calls


def test_apply_stream_options_without_optional_values():
    writer = FakeStreamWriter()

    result = apply_stream_options(
        writer=writer,
    )

    assert result is writer
    assert writer.calls == []
