"""Tests for provider-independent data pipeline orchestration."""

from pathlib import Path

import pytest

from ai_platform.data.contracts import Pipeline
from ai_platform.data.ingestion import LocalFileReader
from ai_platform.data.pipelines import DataPipeline
from ai_platform.data.quality import (
    NotNullRule,
    QualityEngine,
    UniqueRule,
)
from ai_platform.data.sources import DataSourceConfig
from ai_platform.data.transformations import (
    ColumnRenameTransformer,
)


def create_csv(
    tmp_path: Path,
) -> Path:
    """Create a representative customer dataset."""

    file_path = tmp_path / "customers.csv"

    file_path.write_text(
        "id,name\n" "1,Alice\n" "2,Bob\n" "3,Charlie\n",
        encoding="utf-8",
    )

    return file_path


def test_data_pipeline_implements_pipeline_contract(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
            "format": "csv",
        },
    )

    pipeline = DataPipeline(
        name="customer_pipeline",
        source=source,
        reader=LocalFileReader(),
    )

    assert isinstance(pipeline, Pipeline)


def test_data_pipeline_reads_records(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    pipeline = DataPipeline(
        name="customer_pipeline",
        source=source,
        reader=LocalFileReader(),
    )

    result = pipeline.execute()

    assert result["pipeline_name"] == "customer_pipeline"
    assert result["status"] == "completed"
    assert result["records_read"] == 3
    assert result["records_transformed"] == 3
    assert result["quality_passed"] is True


def test_data_pipeline_executes_quality_engine(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    quality_engine = QualityEngine(
        [
            NotNullRule("id"),
            UniqueRule("id"),
        ]
    )

    pipeline = DataPipeline(
        name="customer_quality_pipeline",
        source=source,
        reader=LocalFileReader(),
        quality_engine=quality_engine,
    )

    result = pipeline.execute()

    assert result["quality_passed"] is True
    assert result["quality_report"]["total_rules"] == 2
    assert result["quality_report"]["passed_rules"] == 2
    assert result["quality_report"]["failed_rules"] == 0


def test_data_pipeline_detects_quality_failure(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "customers.csv"

    file_path.write_text(
        "id,name\n" "1,Alice\n" "1,Bob\n",
        encoding="utf-8",
    )

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    quality_engine = QualityEngine(
        [
            UniqueRule("id"),
        ]
    )

    pipeline = DataPipeline(
        name="customer_quality_pipeline",
        source=source,
        reader=LocalFileReader(),
        quality_engine=quality_engine,
    )

    result = pipeline.execute()

    assert result["status"] == "quality_failed"
    assert result["quality_passed"] is False
    assert result["quality_report"]["failed_rules"] == 1


def test_data_pipeline_supports_transformers(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    transformer = ColumnRenameTransformer(
        {
            "id": "customer_id",
        }
    )

    quality_engine = QualityEngine(
        [
            NotNullRule("customer_id"),
            UniqueRule("customer_id"),
        ]
    )

    pipeline = DataPipeline(
        name="customer_transformation_pipeline",
        source=source,
        reader=LocalFileReader(),
        transformers=[transformer],
        quality_engine=quality_engine,
    )

    result = pipeline.execute()

    assert result["status"] == "completed"
    assert result["records_read"] == 3
    assert result["records_transformed"] == 3
    assert result["quality_passed"] is True


def test_data_pipeline_supports_multiple_transformers(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    first_transformer = ColumnRenameTransformer(
        {
            "id": "customer_id",
        }
    )

    second_transformer = ColumnRenameTransformer(
        {
            "name": "customer_name",
        }
    )

    quality_engine = QualityEngine(
        [
            NotNullRule("customer_id"),
            NotNullRule("customer_name"),
        ]
    )

    pipeline = DataPipeline(
        name="multi_transform_pipeline",
        source=source,
        reader=LocalFileReader(),
        transformers=[
            first_transformer,
            second_transformer,
        ],
        quality_engine=quality_engine,
    )

    result = pipeline.execute()

    assert result["status"] == "completed"
    assert result["quality_passed"] is True


def test_data_pipeline_without_quality_engine(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    pipeline = DataPipeline(
        name="simple_pipeline",
        source=source,
        reader=LocalFileReader(),
    )

    result = pipeline.execute()

    assert result["quality_passed"] is True
    assert result["quality_report"]["total_rules"] == 0


def test_data_pipeline_rejects_invalid_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="DataSource protocol",
    ):
        DataPipeline(
            name="invalid_pipeline",
            source=object(),
            reader=LocalFileReader(),
        )


def test_data_pipeline_rejects_invalid_reader(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    with pytest.raises(
        TypeError,
        match="DataReader protocol",
    ):
        DataPipeline(
            name="invalid_pipeline",
            source=source,
            reader=object(),
        )


def test_data_pipeline_requires_name(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    with pytest.raises(
        ValueError,
        match="requires a name",
    ):
        DataPipeline(
            name="",
            source=source,
            reader=LocalFileReader(),
        )


def test_data_pipeline_result_is_serializable(
    tmp_path: Path,
) -> None:
    file_path = create_csv(tmp_path)

    source = DataSourceConfig(
        name="customers",
        source_type="local_file",
        config={
            "path": str(file_path),
        },
    )

    pipeline = DataPipeline(
        name="serialization_pipeline",
        source=source,
        reader=LocalFileReader(),
    )

    result = pipeline.execute()

    assert isinstance(result, dict)
    assert result["pipeline_name"] == ("serialization_pipeline")
    assert isinstance(
        result["quality_report"],
        dict,
    )
