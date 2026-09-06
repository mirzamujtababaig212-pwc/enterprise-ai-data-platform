from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

SUPPORTED_DTYPES: Final[frozenset[str]] = frozenset(
    {
        "int",
        "int64",
        "float",
        "float32",
        "float64",
        "bool",
        "string",
        "category",
    }
)


@dataclass(frozen=True)
class FeatureDefinition:
    """Platform-level definition of a single model feature."""

    name: str
    dtype: str
    required: bool = True
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature name must not be empty")

        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported feature dtype: {self.dtype}")

        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError("min_value must not be greater than max_value")

        object.__setattr__(
            self,
            "allowed_values",
            tuple(self.allowed_values),
        )


@dataclass(frozen=True)
class FeatureContract:
    """Immutable contract describing the features required by a model."""

    name: str
    features: tuple[FeatureDefinition, ...] = ()
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature contract name must not be empty")

        if not self.version.strip():
            raise ValueError("feature contract version must not be empty")

        definitions = tuple(self.features)

        names = [feature.name for feature in definitions]

        if len(names) != len(set(names)):
            raise ValueError("feature contract must not contain duplicate feature names")

        object.__setattr__(self, "features", definitions)

    @property
    def required_features(self) -> tuple[FeatureDefinition, ...]:
        """Return required feature definitions."""

        return tuple(feature for feature in self.features if feature.required)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names in contract order."""

        return tuple(feature.name for feature in self.features)


@dataclass(frozen=True)
class FeatureValidationResult:
    """Result produced by feature-contract validation."""

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    rows_checked: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "errors",
            tuple(self.errors),
        )
        object.__setattr__(
            self,
            "warnings",
            tuple(self.warnings),
        )

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)
