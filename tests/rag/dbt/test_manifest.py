from rag.dbt import DbtManifestParser


def test_manifest_parser_parses_only_models():
    manifest = {
        "nodes": {
            "model.vehicle_dbt.fct_vehicle_metrics": {
                "resource_type": "model",
                "name": "fct_vehicle_metrics",
                "database": "vehicle_platform",
                "schema": "public",
                "alias": "fct_vehicle_metrics",
                "description": "Vehicle metrics",
                "config": {
                    "materialized": "table",
                },
                "tags": ["ai_knowledge"],
                "meta": {
                    "owner": "data-platform",
                },
                "depends_on": {
                    "nodes": [
                        "model.vehicle_dbt.dim_vehicle",
                    ],
                },
                "columns": {
                    "vehicle_id": {
                        "description": "Vehicle identifier",
                        "tags": ["identifier"],
                        "meta": {
                            "pii": False,
                        },
                    },
                },
            },
            "source.vehicle_platform.vehicle_gold": {
                "resource_type": "source",
                "name": "vehicle_gold",
            },
        }
    }

    models = DbtManifestParser().parse(manifest)

    assert len(models) == 1

    model = models[0]

    assert model.unique_id == "model.vehicle_dbt.fct_vehicle_metrics"
    assert model.name == "fct_vehicle_metrics"
    assert model.materialization == "table"
    assert model.tags == ("ai_knowledge",)
    assert model.depends_on == ("model.vehicle_dbt.dim_vehicle",)

    assert len(model.columns) == 1
    assert model.columns[0].name == "vehicle_id"
    assert model.columns[0].description == "Vehicle identifier"
