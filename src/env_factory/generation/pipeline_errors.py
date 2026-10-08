"""Shared task-pipeline exception types."""


class PipelineGenerationError(ValueError):
    """Raised when one pipeline stage cannot produce a valid artifact."""


def is_transient_llm_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in (
        "http 429", "http 502", "http 503", "http 504",
        "too many requests", "service is too busy", "temporarily unavailable",
        "timed out", "timeout", "connection reset",
        "unable to read a complete llm response",
    ))
