class ToolExecutionWaitingForApprovalError(RuntimeError):
    """Raised when tool execution must pause for human approval."""


class ToolExecutionOwnershipLostError(RuntimeError):
    """Raised when a tool execution loses durable run ownership."""
