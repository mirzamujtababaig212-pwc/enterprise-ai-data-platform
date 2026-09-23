class AgentRunStepError(Exception):
    """Base exception for durable agent-run step operations."""


class AgentRunStepNotFoundError(AgentRunStepError):
    """Raised when a requested agent-run step does not exist."""


class DuplicateAgentRunStepError(AgentRunStepError):
    """Raised when an agent-run step already exists."""


class InvalidAgentRunStepTransitionError(AgentRunStepError):
    """Raised when an invalid step lifecycle transition is requested."""
