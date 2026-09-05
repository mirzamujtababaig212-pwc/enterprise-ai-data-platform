from ml.models.customer_churn import (
    DEFAULT_MODEL_PARAMS,
    FEATURE_COLUMNS,
    MODEL_NAME,
    TARGET_COLUMN,
    validate_feature_columns,
)


def test_customer_churn_model_definition() -> None:
    assert MODEL_NAME == "CustomerChurnModel"
    assert TARGET_COLUMN == "churn"

    assert FEATURE_COLUMNS == (
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_tickets",
        "usage_hours",
        "payment_failures",
    )

    assert DEFAULT_MODEL_PARAMS["C"] == 1.0
    assert DEFAULT_MODEL_PARAMS["max_iter"] == 1000
    assert DEFAULT_MODEL_PARAMS["solver"] == "liblinear"


def test_customer_churn_feature_validation() -> None:
    validate_feature_columns(list(FEATURE_COLUMNS))


def test_customer_churn_feature_validation_rejects_missing_columns() -> None:
    try:
        validate_feature_columns(["tenure_months"])
    except ValueError as exc:
        assert "monthly_charges" in str(exc)
    else:
        raise AssertionError("Expected missing feature validation to fail")
