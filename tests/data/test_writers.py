"""Tests for data writer implementations."""

import json
from pathlib import Path

import pytest

from ai_platform.data.contracts import DataWriter
from ai_platform.data.writers import LocalFileWriter


def test_local_file_writer_implements_writer_contract() -> None:
    writer = LocalFileWriter()

    assert isinstance(writer, DataWriter)


def test_local_file_writer_writes_csv(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customers.csv"

    writer = LocalFileWriter()

    result = writer.write(
        [
            {
                "id": 1,
                "name": "Alice",
            },
            {
                "id": 2,
                "name": "Bob",
            },
        ],
        str(destination),
    )

    assert result["status"] == "completed"
    assert result["format"] == "csv"
    assert result["records_written"] == 2

    assert destination.read_text(
        encoding="utf-8",
    ) == ("id,name\n" "1,Alice\n" "2,Bob\n")


def test_local_file_writer_writes_json(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customers.json"

    writer = LocalFileWriter()

    result = writer.write(
        [
            {
                "id": 1,
                "name": "Alice",
            },
            {
                "id": 2,
                "name": "Bob",
            },
        ],
        str(destination),
    )

    assert result["format"] == "json"
    assert result["records_written"] == 2

    payload = json.loads(
        destination.read_text(
            encoding="utf-8",
        )
    )

    assert payload == [
        {
            "id": 1,
            "name": "Alice",
        },
        {
            "id": 2,
            "name": "Bob",
        },
    ]


def test_local_file_writer_writes_jsonl(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customers.jsonl"

    writer = LocalFileWriter()

    result = writer.write(
        [
            {
                "id": 1,
                "name": "Alice",
            },
            {
                "id": 2,
                "name": "Bob",
            },
        ],
        str(destination),
    )

    assert result["format"] == "jsonl"
    assert result["records_written"] == 2

    lines = destination.read_text(
        encoding="utf-8",
    ).splitlines()

    assert [json.loads(line) for line in lines] == [
        {
            "id": 1,
            "name": "Alice",
        },
        {
            "id": 2,
            "name": "Bob",
        },
    ]


def test_local_file_writer_supports_single_record(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customer.json"

    writer = LocalFileWriter()

    result = writer.write(
        {
            "id": 1,
            "name": "Alice",
        },
        str(destination),
    )

    assert result["records_written"] == 1

    payload = json.loads(
        destination.read_text(
            encoding="utf-8",
        )
    )

    assert payload == [
        {
            "id": 1,
            "name": "Alice",
        }
    ]


def test_local_file_writer_creates_parent_directories(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output" / "nested" / "customers.json"

    writer = LocalFileWriter()

    writer.write(
        [
            {"id": 1},
        ],
        str(destination),
    )

    assert destination.exists()


def test_local_file_writer_infers_jsonl_from_ndjson(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customers.ndjson"

    writer = LocalFileWriter()

    result = writer.write(
        [
            {"id": 1},
            {"id": 2},
        ],
        str(destination),
    )

    assert result["format"] == "jsonl"


def test_local_file_writer_rejects_empty_destination() -> None:
    writer = LocalFileWriter()

    with pytest.raises(
        ValueError,
        match="requires a destination",
    ):
        writer.write(
            [{"id": 1}],
            "",
        )


def test_local_file_writer_rejects_unknown_format(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customers.xml"

    writer = LocalFileWriter()

    with pytest.raises(
        ValueError,
        match="Cannot infer supported format",
    ):
        writer.write(
            [{"id": 1}],
            str(destination),
        )


def test_local_file_writer_rejects_invalid_records(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customers.json"

    writer = LocalFileWriter()

    with pytest.raises(
        TypeError,
        match="list of dictionaries",
    ):
        writer.write(
            [
                {"id": 1},
                "invalid",
            ],
            str(destination),
        )


def test_local_file_writer_rejects_unsupported_input(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customers.json"

    writer = LocalFileWriter()

    with pytest.raises(
        TypeError,
        match="dictionary or a list of dictionaries",
    ):
        writer.write(
            "invalid",
            str(destination),
        )


def test_local_file_writer_supports_explicit_format(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "customers.output"

    writer = LocalFileWriter()

    result = writer.write(
        [{"id": 1}],
        str(destination),
        format="json",
    )

    assert result["format"] == "json"

    payload = json.loads(
        destination.read_text(
            encoding="utf-8",
        )
    )

    assert payload == [{"id": 1}]
