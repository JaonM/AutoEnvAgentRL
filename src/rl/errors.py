"""Classify infrastructure failures separately from model actions and truncation."""

class InfrastructureError(RuntimeError):
    pass


class ContextBudgetExceeded(ValueError):
    pass
