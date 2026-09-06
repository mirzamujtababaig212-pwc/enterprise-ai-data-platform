from __future__ import annotations

from ml.platform import FeatureContract, FeatureDefinition

LOAN_DEFAULT_FEATURE_CONTRACT = FeatureContract(
    name="loan-default",
    version="v1",
    features=(
        FeatureDefinition(
            name="income",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="age",
            dtype="int",
            min_value=18,
            max_value=100,
        ),
        FeatureDefinition(
            name="credit_score",
            dtype="float",
            min_value=300,
            max_value=850,
        ),
        FeatureDefinition(
            name="loan_amount",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="employment_years",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="debt_to_income",
            dtype="float",
            min_value=0,
            max_value=1,
        ),
    ),
)
