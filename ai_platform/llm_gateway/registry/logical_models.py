"""Logical model routing definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogicalModelRoute:
    """
    Map a logical model capability to a provider-specific physical model.
    """

    logical_model: str
    capability: str
    provider: str
    model: str
