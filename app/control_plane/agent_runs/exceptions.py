class DuplicateAgentRunError(ValueError):
    """Raised when an agent run with the same ID already exists."""


class AgentRunNotFoundError(LookupError):
    """Raised when an agent run cannot be found for an update."""


class AgentRunAccessDeniedError(PermissionError):
    """Raised when a principal cannot access an agent run."""


class AgentRunIdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused for a different request."""


class InvalidAgentRunTransitionError(ValueError):
    """Raised when an agent run lifecycle transition is not allowed."""


class AgentRunAdmissionRejectedError(RuntimeError):
    """Raised when an agent run is not admitted for execution."""


class AgentRunAlreadyExecutingError(RuntimeError):
    """Raised when another worker already owns execution of an agent run."""


class RecoveryExhaustedError(RuntimeError):
    """Raised when an agent run has exhausted its durable recovery attempts."""
