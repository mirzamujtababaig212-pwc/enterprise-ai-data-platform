from unittest.mock import Mock

from rag.dbt import DbtModel, DbtModelDocumentLoader
from rag.models import Document


def test_dbt_model_document_loader_enriches_documents_with_lineage():
    model = DbtModel(
        unique_id="model.vehicle_dbt.fct_vehicle_metrics",
        name="fct_vehicle_metrics",
        resource_type="model",
        database="vehicle_platform",
        schema="public",
        alias="fct_vehicle_metrics",
        materialization="table",
    )

    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "V001",
                        "avg_speed": 48.17,
                    }
                )
            )
        ]
    )

    reader = Mock()
    reader.read.return_value = dataframe

    loader = DbtModelDocumentLoader(
        model=model,
        reader=reader,
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Vehicle {row['vehicle_id']} " f"average speed {row['avg_speed']}"
        ),
        metadata_fn=lambda row: {
            "source": "vehicle_platform",
            "vehicle_id": row["vehicle_id"],
        },
    )

    spark = Mock()

    documents = loader.load(spark)

    assert documents == [
        Document(
            id="vehicle:V001",
            content="Vehicle V001 average speed 48.17",
            metadata={
                "source": "vehicle_platform",
                "vehicle_id": "V001",
                "dbt_unique_id": "model.vehicle_dbt.fct_vehicle_metrics",
                "dbt_model": "fct_vehicle_metrics",
                "dbt_database": "vehicle_platform",
                "dbt_schema": "public",
                "dbt_alias": "fct_vehicle_metrics",
                "dbt_materialization": "table",
            },
        )
    ]

    reader.read.assert_called_once_with(spark)
