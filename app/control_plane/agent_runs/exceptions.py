class DuplicateAgentRunError(ValueError):
    """Raised when an agent run with the same ID already exists."""


class AgentRunNotFoundError(LookupError):
    """Raised when an agent run cannot be found for an update."""


class InvalidAgentRunTransitionError(ValueError):
    """Raised when an agent run lifecycle transition is not allowed."""


class AgentRunAdmissionRejectedError(RuntimeError):
    """Raised when an agent run is not admitted for execution."""
