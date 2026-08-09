import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

_SESSION_SCRATCH_DIR = None


@pytest.fixture(scope="session", autouse=True)
def test_scratch_base(monkeypatch_session):
    """Provides a single, persistent scratch base directory for the entire session lifecycle

    and globally overrides Python tempfile storage locations to enforce strict data isolation.
    """
    global _SESSION_SCRATCH_DIR
    base_dir = Path(tempfile.mkdtemp(prefix="platform_test_scratch_"))
    _SESSION_SCRATCH_DIR = base_dir

    # Force all down-stream calls to tempfile.mkdtemp or TemporaryDirectory to use this root
    monkeypatch_session.setattr(tempfile, "tempdir", str(base_dir))

    yield base_dir

    # Safely clear out everything only when the full test runner finishes completely
    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Helper fixture to allow session-scoped monkeypatching of core modules."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def spark(test_scratch_base):
    warehouse_path = test_scratch_base / "warehouse"
    metastore_path = test_scratch_base / "metastore_db"

    builder = (
        SparkSession.builder.master("local[*]")
        .appName("EnterprisePlatformTest")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        # Route the metadata warehouse inside the persistent session scratch boundaries
        .config("spark.sql.warehouse.dir", str(warehouse_path))
        # Force an isolated local Derby metastore location per test run
        .config(
            "javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={metastore_path};create=true",
        )
        # Performance tuning to minimize long-running environment footprint
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    yield spark
    spark.stop()


@pytest.fixture
def sample_df(spark):
    return spark.createDataFrame([(1, "CarA"), (2, "CarB")], ["id", "name"])


@pytest.fixture
def mock_reader():
    return Mock()


@pytest.fixture
def mock_writer():
    return Mock()


@pytest.fixture
def mock_validator():
    return Mock()


@pytest.fixture
def mock_metrics():
    return Mock()


@pytest.fixture
def mock_dlq():
    return Mock()


@pytest.fixture
def mock_transformer():
    return Mock()
