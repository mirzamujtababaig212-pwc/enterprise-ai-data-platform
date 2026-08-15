"""Concrete data source configuration models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataSourceConfig:
    """
    Concrete implementation of the DataSource contract.

    The class intentionally contains only source metadata and configuration.
    It does not perform I/O.
    """

    name: str
    source_type: str
    config: dict[str, Any] = field(default_factory=dict)

    def get_config(self) -> dict[str, Any]:
        """Return a defensive copy of the source configuration."""
        return dict(self.config)
