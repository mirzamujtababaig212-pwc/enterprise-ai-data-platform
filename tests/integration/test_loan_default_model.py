from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.models.loan_default import (
    FEATURE_COLUMNS,
    LoanDefaultMLP,
    validate_feature_columns,
)


def test_loan_default_model_architecture() -> None:
    model = LoanDefaultMLP()

    assert model.feature_mean.shape == (len(FEATURE_COLUMNS),)
    assert model.feature_std.shape == (len(FEATURE_COLUMNS),)

    layers = list(model.network)

    assert isinstance(layers[0], torch.nn.Linear)
    assert layers[0].in_features == len(FEATURE_COLUMNS)
    assert layers[0].out_features == 16

    assert isinstance(layers[1], torch.nn.ReLU)

    assert isinstance(layers[2], torch.nn.Linear)
    assert layers[2].in_features == 16
    assert layers[2].out_features == 16

    assert isinstance(layers[3], torch.nn.ReLU)

    assert isinstance(layers[4], torch.nn.Linear)
    assert layers[4].in_features == 16
    assert layers[4].out_features == 1


def test_loan_default_model_forward_pass() -> None:
    model = LoanDefaultMLP()

    inputs = torch.tensor(
        [
            [60000, 35, 720, 20000, 8, 0.25],
            [30000, 52, 540, 25000, 3, 0.55],
        ],
        dtype=torch.float32,
    )

    logits = model(inputs)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_loan_default_feature_validation() -> None:
    validate_feature_columns(list(FEATURE_COLUMNS))


def test_loan_default_missing_feature_rejected() -> None:
    incomplete = list(FEATURE_COLUMNS[:-1])

    with pytest.raises(ValueError, match="missing required"):
        validate_feature_columns(incomplete)


def test_loan_default_feature_normalization_is_persisted() -> None:
    mean = np.arange(len(FEATURE_COLUMNS), dtype=np.float32)
    std = np.arange(1, len(FEATURE_COLUMNS) + 1, dtype=np.float32)

    model = LoanDefaultMLP(
        feature_mean=mean,
        feature_std=std,
    )

    assert np.allclose(
        model.feature_mean.detach().numpy(),
        mean,
    )

    assert np.allclose(
        model.feature_std.detach().numpy(),
        std,
    )
