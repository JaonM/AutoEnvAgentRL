"""Shared task-pipeline exception types."""


class PipelineGenerationError(ValueError):
    """Raised when one pipeline stage cannot produce a valid artifact."""
