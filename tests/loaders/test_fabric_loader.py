from unittest.mock import Mock

from common.fabric.metadata import FabricTableMetadata
from common.readers.fabric_reader import FabricReader
from rag.loaders.fabric import FabricDocumentLoader


def test_fabric_document_loader_adds_canonical_source_provenance():
    metadata = FabricTableMetadata(
        workspace_name="VehicleWorkspace",
        lakehouse_name="VehicleLakehouse",
        catalog_name="VehicleLakehouse.Lakehouse",
        schema_name="dbo",
        table_id="table-123",
        table_name="vehicle_events",
        table_type="MANAGED",
        location="https://onelake.example/Tables/vehicle_events",
        format="DELTA",
    )

    reader = Mock(spec=FabricReader)

    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "VH-001",
                        "avg_speed": 48.17,
                    }
                )
            )
        ]
    )
    reader.read.return_value = dataframe

    spark = Mock()

    documents = FabricDocumentLoader(
        metadata=metadata,
        reader=reader,
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Fabric vehicle {row['vehicle_id']} has an " f"average speed of {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "fabric",
            "dataset": metadata.table_name,
            "workspace": metadata.workspace_name,
            "lakehouse": metadata.lakehouse_name,
            "vehicle_id": row["vehicle_id"],
        },
    ).load(spark)

    reader.read.assert_called_once_with(spark)
    dataframe.toLocalIterator.assert_called_once()

    assert len(documents) == 1
    assert documents[0].id == "vehicle:VH-001"

    assert documents[0].metadata["source"] == "fabric"
    assert documents[0].metadata["dataset"] == "vehicle_events"
    assert documents[0].metadata["workspace"] == "VehicleWorkspace"
    assert documents[0].metadata["lakehouse"] == "VehicleLakehouse"
    assert documents[0].metadata["vehicle_id"] == "VH-001"

    assert documents[0].metadata["source_ref"] == {
        "platform": "fabric",
        "object_type": "table",
        "object_name": "vehicle_events",
        "object_id": "table-123",
        "namespace": "VehicleWorkspace.VehicleLakehouse.dbo",
    }
