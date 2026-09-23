from __future__ import annotations


class MCPServerAlreadyExistsError(RuntimeError):
    """Raised when an MCP server identity or name already exists."""


class MCPServerNotFoundError(LookupError):
    """Raised when a requested MCP server does not exist."""
