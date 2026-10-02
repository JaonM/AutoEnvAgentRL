"""Regression for computed query parameters in multistep reward bindings."""
import copy

import pytest

from env_factory.generation.reward_bindings import bind_reward_queries
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator


def computed_query_source():
    return {
        "tool_implementations": [
            {"tool_name": "read_demand", "operation": "select", "table": "demand",
             "projection": ["name", "boxes"],
             "filters": [{"argument": "name", "column": "name", "operator": "eq"}]},
            {"tool_name": "find_vehicle", "operation": "select", "table": "vehicles",
             "projection": ["name", "capacity"],
             "filters": [{"argument": "capacity", "column": "capacity", "operator": "eq"}]},
        ],
        "scenarios": [{"kind": "goal_success", "steps": [
            {"operation": "tool_call", "tool_name": "read_demand",
             "arguments": {"name": "east"}, "capture": {"boxes": "$.records[0].boxes"}},
            {"operation": "tool_call", "tool_name": "find_vehicle", "arguments": {
                "capacity": {"$expr": {"op": "max", "args": [10,
                    {"op": "add", "args": [{"$ref": "boxes"}, 2]}]}}}},
        ]}],
        "metric_implementations": [{"operator": "value_targets", "expected": {
            "targets": [{"key": "vehicle", "expression": {
                "from_tool": {"name": "find_vehicle", "field": "name"}}}]}}],
    }


def test_computed_query_reward_follows_changed_upstream_business_data():
    source = computed_query_source()
    bind_reward_queries(source)
    expression = source["metric_implementations"][0]["expected"]["targets"][0]["expression"]
    for boxes, expected in [(5, "small"), (12, "large")]:
        argument = computed_query_source()["scenarios"][0]["steps"][1]["arguments"]["capacity"]
        assert DeclarativeMetricEvaluator.resolve_capture_argument(argument, {"boxes": boxes}) == max(10, boxes + 2)
        state = {"demand": [{"name": "east", "boxes": boxes}], "vehicles": [
            {"name": "small", "capacity": 10}, {"name": "large", "capacity": 14}]}
        assert DeclarativeMetricEvaluator._value_expression(expression, state) == expected
    # Missing upstream evidence cannot become a constant target or a passing answer.
    state["demand"] = []
    assert DeclarativeMetricEvaluator._value_expression(expression, state) is None


def test_computed_query_with_unknown_capture_is_rejected_at_binding():
    source = copy.deepcopy(computed_query_source())
    source["scenarios"][0]["steps"][0]["capture"] = {}
    with pytest.raises(ValueError, match="REWARD_QUERY_BINDING"):
        bind_reward_queries(source)


def test_nested_expression_wrapper_is_rejected_like_the_reference_interpreter():
    source = computed_query_source()
    expression = source["scenarios"][0]["steps"][1]["arguments"]["capacity"]["$expr"]
    expression["args"][1] = {"$expr": expression["args"][1]}
    with pytest.raises(ValueError, match="REWARD_QUERY_BINDING"):
        bind_reward_queries(source)
