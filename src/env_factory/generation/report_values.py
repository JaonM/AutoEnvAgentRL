"""Convert computed diagnostic evidence to exact, JSON-safe values."""

from decimal import Decimal


def report_value(value):
    """Normalize reports only; never use these tagged values as DSL operands."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError('REPORT_VALUE_INVALID: non-finite computed decimal')
        return int(value) if value == value.to_integral_value() else {'decimal': str(value)}
    if isinstance(value, dict):
        return {key: report_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [report_value(child) for child in value]
    return value
