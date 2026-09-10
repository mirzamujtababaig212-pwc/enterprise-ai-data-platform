from rag.dbt import DbtModel, DbtModelSelectionPolicy


def _model(
    name: str,
    *,
    tags: tuple[str, ...] = (),
) -> DbtModel:
    return DbtModel(
        unique_id=f"model.vehicle_dbt.{name}",
        name=name,
        resource_type="model",
        tags=tags,
    )


def test_reporting_model_is_selected():
    policy = DbtModelSelectionPolicy()

    assert policy.select(_model("rpt_vehicle_summary")) is True


def test_fact_model_is_selected():
    policy = DbtModelSelectionPolicy()

    assert policy.select(_model("fct_vehicle_metrics")) is True


def test_dimension_model_is_selected():
    policy = DbtModelSelectionPolicy()

    assert policy.select(_model("dim_vehicle")) is True


def test_intermediate_model_is_not_selected_by_default():
    policy = DbtModelSelectionPolicy()

    assert policy.select(_model("int_vehicle_health")) is False


def test_staging_model_is_not_selected_by_default():
    policy = DbtModelSelectionPolicy()

    assert policy.select(_model("stg_vehicle_gold")) is False


def test_test_model_is_excluded():
    policy = DbtModelSelectionPolicy()

    assert policy.select(_model("dbt_spark_metastore_test")) is False


def test_ai_knowledge_tag_selects_model():
    policy = DbtModelSelectionPolicy()

    assert (
        policy.select(
            _model(
                "int_vehicle_health",
                tags=("ai_knowledge",),
            )
        )
        is True
    )


def test_ai_exclude_tag_overrides_selection():
    policy = DbtModelSelectionPolicy()

    assert (
        policy.select(
            _model(
                "rpt_vehicle_summary",
                tags=("ai_exclude",),
            )
        )
        is False
    )
