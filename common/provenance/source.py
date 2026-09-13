from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnterpriseSourceRef:
    """
    Canonical cross-platform reference to an enterprise data object.

    Platform-specific metadata remains authoritative in the corresponding
    platform metadata model. This class only provides a small, stable
    identity contract suitable for propagation through RAG metadata.
    """

    platform: str
    object_type: str
    object_name: str | None = None
    object_id: str | None = None
    namespace: str | None = None
    environment: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if not self.platform.strip():
            raise ValueError("Source platform must be a non-empty string.")

        if not self.object_type.strip():
            raise ValueError("Source object type must be a non-empty string.")

        if self.object_name is None and self.object_id is None:
            raise ValueError("Source reference requires object_name or object_id.")

        if self.object_name is not None and not self.object_name.strip():
            raise ValueError("Source object name must be non-empty when provided.")

        if self.object_id is not None and not self.object_id.strip():
            raise ValueError("Source object ID must be non-empty when provided.")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation without null fields."""
        values = {
            "platform": self.platform,
            "object_type": self.object_type,
            "object_name": self.object_name,
            "object_id": self.object_id,
            "namespace": self.namespace,
            "environment": self.environment,
            "version": self.version,
        }
        return {key: value for key, value in values.items() if value is not None}
