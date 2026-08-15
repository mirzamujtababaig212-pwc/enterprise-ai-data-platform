"""
Data Platform Protocols & Abstractions.

These protocols define the architectural contracts between data-platform
components without coupling the contracts to a specific implementation,
framework, cloud provider, or storage engine.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSource(Protocol):
    """Contract describing a logical data source."""

    name: str
    source_type: str

    def get_config(self) -> dict[str, Any]:
        """Return source configuration."""
        ...


@runtime_checkable
class DataReader(Protocol):
    """Contract for components that read data."""

    def read(
        self,
        source: DataSource,
        **kwargs: Any,
    ) -> Any:
        """Read data from a source."""
        ...


@runtime_checkable
class DataWriter(Protocol):
    """Contract for components that write data."""

    def write(
        self,
        data: Any,
        destination: str,
        **kwargs: Any,
    ) -> Any:
        """Write data to a destination."""
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Contract for data transformation components."""

    def transform(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Transform input data."""
        ...


@runtime_checkable
class DataQualityRule(Protocol):
    """Contract for data-quality rules."""

    rule_name: str

    def evaluate(
        self,
        data: Any,
    ) -> dict[str, Any]:
        """Evaluate data against the quality rule."""
        ...


@runtime_checkable
class Pipeline(Protocol):
    """Contract for executable data pipelines."""

    name: str

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the pipeline."""
        ...
