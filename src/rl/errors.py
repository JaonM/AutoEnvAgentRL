"""Classify infrastructure failures separately from model actions and truncation."""

class InfrastructureError(RuntimeError):
    pass


class ContextBudgetExceeded(ValueError):
    pass


class EnvironmentProtocolError(RuntimeError):
    """Non-retryable environment RPC failure; never a policy failure reward."""
