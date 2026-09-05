from __future__ import annotations

import pandas as pd
import pytest

from ml.inference import LoanDefaultPredictor


@pytest.fixture
def loan_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "income": 60000.0,
                "age": 35,
                "credit_score": 720,
                "loan_amount": 20000.0,
                "employment_years": 8,
                "debt_to_income": 0.25,
            }
        ]
    )


def test_loan_default_champion_inference(
    loan_features: pd.DataFrame,
) -> None:
    predictor = LoanDefaultPredictor(
        model_name="LoanDefaultModel",
        model_alias="champion",
    )

    predictor.load()

    result = predictor.predict(loan_features)

    assert result.default in (0, 1)
    assert 0.0 <= result.default_probability <= 1.0
    assert result.model_name == "LoanDefaultModel"
    assert result.model_alias == "champion"


def test_loan_default_batch_inference(
    loan_features: pd.DataFrame,
) -> None:
    predictor = LoanDefaultPredictor(
        model_name="LoanDefaultModel",
        model_alias="champion",
    )

    result = predictor.predict_batch(
        pd.concat(
            [loan_features, loan_features],
            ignore_index=True,
        )
    )

    assert len(result) == 2
    assert "default" in result.columns
    assert "default_probability" in result.columns
    assert "model_name" in result.columns
    assert "model_alias" in result.columns


def test_loan_default_missing_feature_rejected() -> None:
    predictor = LoanDefaultPredictor(
        model_name="LoanDefaultModel",
        model_alias="champion",
    )

    incomplete = pd.DataFrame(
        [
            {
                "income": 60000.0,
                "age": 35,
            }
        ]
    )

    with pytest.raises(ValueError):
        predictor.predict(incomplete)


def test_loan_default_null_feature_rejected(
    loan_features: pd.DataFrame,
) -> None:
    predictor = LoanDefaultPredictor(
        model_name="LoanDefaultModel",
        model_alias="champion",
    )

    loan_features.loc[0, "credit_score"] = None

    with pytest.raises(ValueError, match="null"):
        predictor.predict(loan_features)
