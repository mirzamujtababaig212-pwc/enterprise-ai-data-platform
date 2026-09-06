from __future__ import annotations

from typing import Any

import pandas as pd

from .features import (
    FeatureContract,
    FeatureDefinition,
    FeatureValidationResult,
)


class FeatureValidator:
    """Validate pandas DataFrames against platform feature contracts."""

    @classmethod
    def validate(
        cls,
        dataframe: pd.DataFrame,
        contract: FeatureContract,
    ) -> FeatureValidationResult:
        """Validate a DataFrame against a feature contract."""

        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(dataframe, pd.DataFrame):
            return FeatureValidationResult(
                valid=False,
                errors=("dataframe must be a pandas DataFrame",),
                rows_checked=0,
            )

        rows_checked = len(dataframe)

        if dataframe.empty:
            errors.append("dataframe must not be empty")

        for feature in contract.required_features:
            if feature.name not in dataframe.columns:
                errors.append(f"missing required feature: {feature.name}")

        contract_names = set(contract.feature_names)

        for column in dataframe.columns:
            if column not in contract_names:
                warnings.append(f"unexpected feature column: {column}")

        for feature in contract.features:
            if feature.name not in dataframe.columns:
                continue

            series = dataframe[feature.name]

            cls._validate_dtype(
                series,
                feature,
                errors,
            )

            cls._validate_nullability(
                series,
                feature,
                errors,
            )

            cls._validate_range(
                series,
                feature,
                errors,
            )

            cls._validate_allowed_values(
                series,
                feature,
                errors,
            )

        return FeatureValidationResult(
            valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            rows_checked=rows_checked,
        )

    @staticmethod
    def _validate_dtype(
        series: pd.Series,
        feature: FeatureDefinition,
        errors: list[str],
    ) -> None:
        dtype = feature.dtype

        if dtype in {"int", "int64"}:
            valid = pd.api.types.is_integer_dtype(series)

        elif dtype == "float":
            valid = pd.api.types.is_numeric_dtype(series)

        elif dtype in {"float32", "float64"}:
            valid = pd.api.types.is_float_dtype(series)

        elif dtype == "bool":
            valid = pd.api.types.is_bool_dtype(series)

        elif dtype == "string":
            valid = pd.api.types.is_string_dtype(series)

        elif dtype == "category":
            valid = isinstance(series.dtype, pd.CategoricalDtype)

        else:
            valid = False

        if not valid:
            errors.append(
                f"feature '{feature.name}' must have dtype "
                f"compatible with '{dtype}', got '{series.dtype}'"
            )

    @staticmethod
    def _validate_nullability(
        series: pd.Series,
        feature: FeatureDefinition,
        errors: list[str],
    ) -> None:
        if not feature.nullable and series.isnull().any():
            null_count = int(series.isnull().sum())

            errors.append(f"feature '{feature.name}' contains " f"{null_count} null value(s)")

    @staticmethod
    def _validate_range(
        series: pd.Series,
        feature: FeatureDefinition,
        errors: list[str],
    ) -> None:
        if not pd.api.types.is_numeric_dtype(series):
            return

        if feature.min_value is not None:
            below_min = series < feature.min_value

            if below_min.any():
                count = int(below_min.sum())

                errors.append(
                    f"feature '{feature.name}' contains "
                    f"{count} value(s) below minimum "
                    f"{feature.min_value}"
                )

        if feature.max_value is not None:
            above_max = series > feature.max_value

            if above_max.any():
                count = int(above_max.sum())

                errors.append(
                    f"feature '{feature.name}' contains "
                    f"{count} value(s) above maximum "
                    f"{feature.max_value}"
                )

    @staticmethod
    def _validate_allowed_values(
        series: pd.Series,
        feature: FeatureDefinition,
        errors: list[str],
    ) -> None:
        if not feature.allowed_values:
            return

        allowed = set(feature.allowed_values)

        invalid_mask = ~series.isin(allowed)

        if invalid_mask.any():
            invalid_values: list[Any] = series[invalid_mask].dropna().unique().tolist()

            errors.append(
                f"feature '{feature.name}' contains "
                f"values outside allowed set: {invalid_values}"
            )
