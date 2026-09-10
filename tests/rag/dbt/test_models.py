from dataclasses import FrozenInstanceError

import pytest

from rag.dbt import DbtColumn, DbtModel


def test_dbt_model_stores_metadata():
    model = DbtModel(
        unique_id="model.vehicle_dbt.fct_vehicle_metrics",
        name="fct_vehicle_metrics",
        resource_type="model",
        database="vehicle_platform",
        schema="public",
        alias="fct_vehicle_metrics",
        description="Vehicle metrics fact table",
        materialization="table",
        tags=("ai_knowledge",),
        depends_on=("model.vehicle_dbt.dim_vehicle",),
        columns=(
            DbtColumn(
                name="vehicle_id",
                description="Vehicle identifier",
            ),
        ),
    )

    assert model.name == "fct_vehicle_metrics"
    assert model.materialization == "table"
    assert model.tags == ("ai_knowledge",)
    assert len(model.columns) == 1


def test_dbt_model_is_immutable():
    model = DbtModel(
        unique_id="model.vehicle_dbt.dim_vehicle",
        name="dim_vehicle",
        resource_type="model",
    )

    with pytest.raises(FrozenInstanceError):
        model.name = "changed"
