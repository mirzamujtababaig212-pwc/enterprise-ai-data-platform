"""Local file data reader implementation."""

import csv
import json
from pathlib import Path
from typing import Any

from ai_platform.data.contracts import DataSource
from ai_platform.data.ingestion.base import BaseDataReader


class LocalFileReader(BaseDataReader):
    """Read structured data from local files."""

    SUPPORTED_FORMATS = {
        "csv",
        "json",
        "jsonl",
    }

    def read(
        self,
        source: DataSource,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Read records from a local file.

        The source configuration must contain:
            path: path to the local file

        Optional:
            format: csv, json, or jsonl

        If format is omitted, it is inferred from the file extension.
        """

        config = source.get_config()

        path_value = config.get("path")

        if not path_value:
            raise ValueError("Local file source requires a 'path' configuration.")

        path = Path(path_value)

        if not path.exists():
            raise FileNotFoundError(f"Data source file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Data source path is not a file: {path}")

        file_format = config.get("format")

        if file_format is None:
            file_format = self._infer_format(path)

        file_format = str(file_format).lower().strip()

        if file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported local file format: {file_format}. "
                f"Supported formats: {sorted(self.SUPPORTED_FORMATS)}"
            )

        if file_format == "csv":
            return self._read_csv(path)

        if file_format == "json":
            return self._read_json(path)

        return self._read_jsonl(path)

    @staticmethod
    def _infer_format(path: Path) -> str:
        """Infer supported format from the file extension."""

        suffix = path.suffix.lower()

        mapping = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
            ".ndjson": "jsonl",
        }

        try:
            return mapping[suffix]
        except KeyError as exc:
            raise ValueError(
                f"Cannot infer supported format from file extension: " f"{suffix or '<none>'}"
            ) from exc

    @staticmethod
    def _read_csv(
        path: Path,
    ) -> list[dict[str, Any]]:
        """Read a CSV file into records."""

        with path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            return [dict(row) for row in reader]

    @staticmethod
    def _read_json(
        path: Path,
    ) -> list[dict[str, Any]]:
        """Read a JSON document containing an object or list of objects."""

        with path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if isinstance(payload, dict):
            return [payload]

        if isinstance(payload, list):
            if not all(isinstance(item, dict) for item in payload):
                raise ValueError("JSON array must contain only objects.")

            return payload

        raise ValueError("JSON source must contain an object or an array of objects.")

    @staticmethod
    def _read_jsonl(
        path: Path,
    ) -> list[dict[str, Any]]:
        """Read newline-delimited JSON."""

        records: list[dict[str, Any]] = []

        with path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} " f"in {path}") from exc

                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL line {line_number} must contain " "an object.")

                records.append(payload)

        return records
