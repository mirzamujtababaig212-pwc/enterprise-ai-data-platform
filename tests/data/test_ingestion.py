"""Tests for data ingestion implementations."""

import json
from pathlib import Path

import pytest

from ai_platform.data.contracts import DataReader
from ai_platform.data.ingestion import (
    BaseDataReader,
    LocalFileReader,
)
from ai_platform.data.sources import DataSourceConfig


def test_local_file_reader_implements_reader_contract() -> None:
    reader = LocalFileReader()

    assert isinstance(reader, DataReader)
    assert isinstance(reader, BaseDataReader)


def test_local_file_reader_reads_csv(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "customers.csv"

    file_path.write_text(
        "id,name\n" "1,Alice\n" "2,Bob\n",
        encoding="utf-8",
    )

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
            "format": "csv",
        },
    )

    records = LocalFileReader().read(source)

    assert records == [
        {
            "id": "1",
            "name": "Alice",
        },
        {
            "id": "2",
            "name": "Bob",
        },
    ]


def test_local_file_reader_reads_json(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "customers.json"

    file_path.write_text(
        json.dumps(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        ),
        encoding="utf-8",
    )

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
            "format": "json",
        },
    )

    records = LocalFileReader().read(source)

    assert records == [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]


def test_local_file_reader_reads_jsonl(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "customers.jsonl"

    file_path.write_text(
        '{"id": 1, "name": "Alice"}\n' '{"id": 2, "name": "Bob"}\n',
        encoding="utf-8",
    )

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
            "format": "jsonl",
        },
    )

    records = LocalFileReader().read(source)

    assert records == [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]


def test_local_file_reader_infers_csv_format(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "customers.csv"

    file_path.write_text(
        "id,name\n" "1,Alice\n",
        encoding="utf-8",
    )

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    records = LocalFileReader().read(source)

    assert records == [
        {
            "id": "1",
            "name": "Alice",
        }
    ]


def test_local_file_reader_requires_path() -> None:
    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
    )

    with pytest.raises(
        ValueError,
        match="requires a 'path'",
    ):
        LocalFileReader().read(source)


def test_local_file_reader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "missing.csv"

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        LocalFileReader().read(source)


def test_local_file_reader_rejects_directory(
    tmp_path: Path,
) -> None:
    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(tmp_path),
        },
    )

    with pytest.raises(
        ValueError,
        match="is not a file",
    ):
        LocalFileReader().read(source)


def test_local_file_reader_rejects_unknown_format(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "customers.xml"

    file_path.write_text(
        "<customers />",
        encoding="utf-8",
    )

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
            "format": "xml",
        },
    )

    with pytest.raises(
        ValueError,
        match="Unsupported local file format",
    ):
        LocalFileReader().read(source)


def test_local_file_reader_rejects_invalid_jsonl(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "customers.jsonl"

    file_path.write_text(
        '{"id": 1}\n' "not-valid-json\n",
        encoding="utf-8",
    )

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON on line 2",
    ):
        LocalFileReader().read(source)


def test_local_file_reader_rejects_non_object_json_array(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "invalid.json"

    file_path.write_text(
        json.dumps([1, 2, 3]),
        encoding="utf-8",
    )

    source = DataSourceConfig(
        name="invalid",
        source_type="local_file",
        config={
            "path": str(file_path),
            "format": "json",
        },
    )

    with pytest.raises(
        ValueError,
        match="must contain only objects",
    ):
        LocalFileReader().read(source)
