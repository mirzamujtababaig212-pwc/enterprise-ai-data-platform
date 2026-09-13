import os

import pytest

from common.databricks.control_plane import DatabricksControlPlaneClient
from common.provenance import databricks_source_ref

pytestmark = pytest.mark.databricks


def test_live_databricks_table_metadata_preserves_source_provenance():
    if os.getenv("RUN_DATABRICKS_INTEGRATION") != "1":
        pytest.skip("Set RUN_DATABRICKS_INTEGRATION=1 to run the Databricks integration test")

    table_name = os.getenv("DATABRICKS_TEST_TABLE")
    if not table_name:
        pytest.skip("Set DATABRICKS_TEST_TABLE to an accessible Databricks table")

    control_plane = DatabricksControlPlaneClient()
    metadata = control_plane.get_table_metadata(table_name)

    assert metadata.full_name
    assert metadata.catalog
    assert metadata.schema
    assert metadata.name

    source_ref = databricks_source_ref(metadata)

    assert source_ref.platform == "databricks"
    assert source_ref.object_type == "table"
    assert source_ref.object_name
    assert source_ref.namespace == f"{metadata.catalog}.{metadata.schema}"
