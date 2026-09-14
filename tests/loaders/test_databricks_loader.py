from unittest.mock import Mock

from common.databricks.metadata import DatabricksTableMetadata
from common.readers.databricks_reader import DatabricksReader
from rag.loaders.databricks import DatabricksDocumentLoader
from rag.models import Document


def test_databricks_document_loader_adds_canonical_source_provenance():
    metadata = DatabricksTableMetadata(
        full_name="vehicle_platform.public.vehicle_events",
        catalog="vehicle_platform",
        schema="public",
        name="vehicle_events",
        table_type="MANAGED",
        table_id="table-123",
    )

    reader = Mock(spec=DatabricksReader)
    reader.read.return_value = Mock()

    loader = DatabricksDocumentLoader(
        metadata=metadata,
        reader=reader,
        id_fn=lambda values: values["vehicle_id"],
        content_fn=lambda values: (
            f"Vehicle {values['vehicle_id']} " f"is {values['status']} in {values['region']}"
        ),
        metadata_fn=lambda values: {
            "source": "databricks",
            "dataset": "vehicle_events",
            "vehicle_id": values["vehicle_id"],
        },
    )

    loader._document_loader.load = Mock(
        return_value=[
            Document(
                id="VH-001",
                content="Vehicle VH-001 is ACTIVE in HYD",
                metadata={
                    "source": "databricks",
                    "dataset": "vehicle_events",
                    "vehicle_id": "VH-001",
                },
            )
        ]
    )

    documents = loader.load(Mock())

    assert len(documents) == 1
    assert documents[0].id == "VH-001"
    assert documents[0].metadata["source"] == "databricks"
    assert documents[0].metadata["dataset"] == "vehicle_events"
    assert documents[0].metadata["vehicle_id"] == "VH-001"

    assert documents[0].metadata["source_ref"] == {
        "platform": "databricks",
        "object_type": "table",
        "object_name": "vehicle_events",
        "object_id": "table-123",
        "namespace": "vehicle_platform.public",
    }

    reader.read.assert_called_once()
