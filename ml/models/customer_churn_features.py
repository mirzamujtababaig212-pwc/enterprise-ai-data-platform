from __future__ import annotations

from ml.platform import FeatureContract, FeatureDefinition

CUSTOMER_CHURN_FEATURE_CONTRACT = FeatureContract(
    name="customer-churn",
    version="v1",
    features=(
        FeatureDefinition(
            name="tenure_months",
            dtype="int",
            min_value=0,
        ),
        FeatureDefinition(
            name="monthly_charges",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="total_charges",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="support_tickets",
            dtype="int",
            min_value=0,
        ),
        FeatureDefinition(
            name="usage_hours",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="payment_failures",
            dtype="int",
            min_value=0,
        ),
    ),
)
