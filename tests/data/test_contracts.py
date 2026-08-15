from typing import Any

from ai_platform.data.contracts import (
    DataQualityRule,
    DataReader,
    DataSource,
    DataTransformer,
    DataWriter,
    Pipeline,
)


class DummySource:
    name = "dummy_source"
    source_type = "local"

    def get_config(self) -> dict[str, Any]:
        return {
            "path": "/tmp/example.csv",
        }


class DummyReader:
    def read(
        self,
        source: DataSource,
        **kwargs: Any,
    ) -> Any:
        return {
            "status": "success",
            "data": [1, 2, 3],
        }


class DummyWriter:
    def write(
        self,
        data: Any,
        destination: str,
        **kwargs: Any,
    ) -> Any:
        return {
            "written": True,
            "destination": destination,
        }


class DummyTransformer:
    def transform(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        return [value * 2 for value in data]


class DummyQualityRule:
    rule_name = "not_null"

    def evaluate(
        self,
        data: Any,
    ) -> dict[str, Any]:
        return {
            "rule": self.rule_name,
            "passed": True,
        }


class DummyPipeline:
    name = "dummy_pipeline"

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "pipeline": self.name,
            "status": "completed",
        }


def test_data_source_protocol() -> None:
    source = DummySource()

    assert isinstance(source, DataSource)
    assert source.name == "dummy_source"
    assert source.source_type == "local"
    assert source.get_config() == {
        "path": "/tmp/example.csv",
    }


def test_data_reader_protocol() -> None:
    reader = DummyReader()
    source = DummySource()

    assert isinstance(reader, DataReader)

    result = reader.read(source)

    assert result["status"] == "success"
    assert result["data"] == [1, 2, 3]


def test_data_writer_protocol() -> None:
    writer = DummyWriter()

    assert isinstance(writer, DataWriter)

    result = writer.write(
        [1, 2],
        "local://output",
    )

    assert result["written"] is True
    assert result["destination"] == "local://output"


def test_data_transformer_protocol() -> None:
    transformer = DummyTransformer()

    assert isinstance(transformer, DataTransformer)

    result = transformer.transform([1, 2, 3])

    assert result == [2, 4, 6]


def test_data_quality_rule_protocol() -> None:
    rule = DummyQualityRule()

    assert isinstance(rule, DataQualityRule)

    result = rule.evaluate([1, 2])

    assert result["rule"] == "not_null"
    assert result["passed"] is True


def test_pipeline_protocol() -> None:
    pipeline = DummyPipeline()

    assert isinstance(pipeline, Pipeline)

    result = pipeline.execute()

    assert result["pipeline"] == "dummy_pipeline"
    assert result["status"] == "completed"
