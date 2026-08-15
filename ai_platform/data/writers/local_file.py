"""Local file writer implementation."""

import csv
import json
from pathlib import Path
from typing import Any


class LocalFileWriter:
    """Write structured records to local files."""

    SUPPORTED_FORMATS = {
        "csv",
        "json",
        "jsonl",
    }

    def write(
        self,
        data: Any,
        destination: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Write records to a local file.

        Supported formats:

        - csv
        - json
        - jsonl

        The format can be supplied through kwargs:

            format="csv"

        If format is omitted, it is inferred from the destination
        file extension.
        """

        if not destination:
            raise ValueError("Local file writer requires a destination.")

        path = Path(destination)

        file_format = kwargs.get("format")

        if file_format is None:
            file_format = self._infer_format(path)

        file_format = str(file_format).lower().strip()

        if file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported local file format: {file_format}. "
                f"Supported formats: "
                f"{sorted(self.SUPPORTED_FORMATS)}"
            )

        records = self._normalize_records(data)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if file_format == "csv":
            self._write_csv(path, records)
        elif file_format == "json":
            self._write_json(path, records)
        else:
            self._write_jsonl(path, records)

        return {
            "destination": str(path),
            "format": file_format,
            "records_written": len(records),
            "status": "completed",
        }

    @staticmethod
    def _infer_format(path: Path) -> str:
        """Infer output format from the destination extension."""

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
                "Cannot infer supported format from file extension: " f"{suffix or '<none>'}"
            ) from exc

    @staticmethod
    def _normalize_records(
        data: Any,
    ) -> list[dict[str, Any]]:
        """Normalize supported input structures into records."""

        if isinstance(data, dict):
            return [dict(data)]

        if isinstance(data, list):
            records: list[dict[str, Any]] = []

            for record in data:
                if not isinstance(record, dict):
                    raise TypeError("LocalFileWriter expects a list of dictionaries.")

                records.append(dict(record))

            return records

        raise TypeError("LocalFileWriter expects a dictionary or " "a list of dictionaries.")

    @staticmethod
    def _write_csv(
        path: Path,
        records: list[dict[str, Any]],
    ) -> None:
        """Write records as CSV."""

        if not records:
            path.write_text(
                "",
                encoding="utf-8",
            )
            return

        fieldnames: list[str] = []

        for record in records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)

        with path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _write_json(
        path: Path,
        records: list[dict[str, Any]],
    ) -> None:
        """Write records as a JSON array."""

        with path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                records,
                file,
                indent=2,
            )

    @staticmethod
    def _write_jsonl(
        path: Path,
        records: list[dict[str, Any]],
    ) -> None:
        """Write records as newline-delimited JSON."""

        with path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            for record in records:
                file.write(json.dumps(record) + "\n")
