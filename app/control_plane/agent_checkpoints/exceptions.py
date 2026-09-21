from __future__ import annotations


class AgentCheckpointOwnershipLostError(RuntimeError):
    """Raised when a worker cannot persist a checkpoint because it lost run ownership."""
