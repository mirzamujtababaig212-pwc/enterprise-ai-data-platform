"""Provider-independent data pipeline orchestration."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ai_platform.data.contracts import (
    DataReader,
    DataSource,
    DataTransformer,
)
from ai_platform.data.quality import QualityEngine


@dataclass(frozen=True)
class PipelineResult:
    """Serializable result produced by a data pipeline."""

    pipeline_name: str
    status: str
    records_read: int
    records_transformed: int
    quality_passed: bool
    quality_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the pipeline result as a dictionary."""

        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status,
            "records_read": self.records_read,
            "records_transformed": self.records_transformed,
            "quality_passed": self.quality_passed,
            "quality_report": dict(self.quality_report),
        }


class DataPipeline:
    """Execute ingestion, transformation, and quality validation."""

    def __init__(
        self,
        name: str,
        source: DataSource,
        reader: DataReader,
        transformers: Iterable[DataTransformer] | None = None,
        quality_engine: QualityEngine | None = None,
    ) -> None:
        if not name:
            raise ValueError("DataPipeline requires a name.")

        if not isinstance(source, DataSource):
            raise TypeError("source must implement the DataSource protocol.")

        if not isinstance(reader, DataReader):
            raise TypeError("reader must implement the DataReader protocol.")

        self.name = name
        self.source = source
        self.reader = reader
        self.transformers = list(transformers or [])
        self.quality_engine = quality_engine

        for transformer in self.transformers:
            if not isinstance(transformer, DataTransformer):
                raise TypeError("All transformers must implement " "the DataTransformer protocol.")

        if quality_engine is not None and not isinstance(
            quality_engine,
            QualityEngine,
        ):
            raise TypeError("quality_engine must be a QualityEngine instance.")

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the complete data pipeline."""

        records = self.reader.read(
            self.source,
            **kwargs,
        )

        records_read = len(records)

        transformed_records = records

        for transformer in self.transformers:
            transformed_records = transformer.transform(
                transformed_records,
                **kwargs,
            )

        records_transformed = len(transformed_records)

        if self.quality_engine is None:
            quality_report = {
                "overall_passed": True,
                "total_rules": 0,
                "passed_rules": 0,
                "failed_rules": 0,
                "results": [],
            }
        else:
            quality_report = self.quality_engine.evaluate(
                transformed_records,
            )

        quality_passed = bool(quality_report["overall_passed"])

        status = "completed" if quality_passed else "quality_failed"

        result = PipelineResult(
            pipeline_name=self.name,
            status=status,
            records_read=records_read,
            records_transformed=records_transformed,
            quality_passed=quality_passed,
            quality_report=quality_report,
        )

        return result.to_dict()
