"""Regression for computed query parameters in multistep reward bindings."""
import copy

import pytest

from env_factory.generation.reward_bindings import bind_reward_queries
from env_factory.generation.reward_bindings import reward_temporal_diagnostics
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


def test_cached_read_retry_can_introduce_downstream_capture():
    source = computed_query_source()
    steps = source['scenarios'][0]['steps']
    steps[0]['idempotency_key'] = 'read-once'
    retry = copy.deepcopy(steps[0])
    steps[0]['capture'] = {}
    steps.insert(1, retry)
    bind_reward_queries(source)
    expression = source['metric_implementations'][0]['expected']['targets'][0]['expression']
    state = {'demand': [{'name': 'east', 'boxes': 12}],
             'vehicles': [{'name': 'large', 'capacity': 14}]}
    assert DeclarativeMetricEvaluator._value_expression(expression, state) == 'large'


@pytest.mark.parametrize('keys', [(None, None), ('a', 'b'), ('a', 'b', 'a')])
def test_distinct_read_requests_remain_ambiguous(keys):
    source = computed_query_source()
    steps = source['scenarios'][0]['steps']
    read = steps[-1]
    for key in keys:
        retry = copy.deepcopy(read)
        if key is not None:
            retry['idempotency_key'] = key
        steps.append(retry)
    steps.remove(read)
    with pytest.raises(ValueError, match='REWARD_QUERY_BINDING'):
        bind_reward_queries(source)


def test_nested_expression_wrapper_is_rejected_like_the_reference_interpreter():
    source = computed_query_source()
    expression = source["scenarios"][0]["steps"][1]["arguments"]["capacity"]["$expr"]
    expression["args"][1] = {"$expr": expression["args"][1]}
    with pytest.raises(ValueError, match="REWARD_QUERY_BINDING"):
        bind_reward_queries(source)


def test_state_predicate_observability_does_not_call_unbound_lowering():
    from env_factory.generation.reward_bindings import validate_reward_observability
    source = computed_query_source()
    source['environment_plan'] = {'mode': 'reference_data'}
    source['tables'] = [{'table_name': 'demand', 'columns': [{'name': 'name'}, {'name': 'boxes'}]},
                        {'table_name': 'vehicles', 'columns': [{'name': 'name'}, {'name': 'capacity'}]}]
    source['metric_implementations'] = [{'operator': 'state_predicates', 'expected': [
        {'table': 'vehicles', 'where': {'name': 'A'}, 'value_expressions': {
            'capacity': {'lookup': {'table': 'demand', 'field': 'boxes', 'where': {'name': 'east'}}}}}]}]
    validate_reward_observability(source)
    source['metric_implementations'][0]['expected'][0]['value_expressions']['capacity']['lookup']['field'] = 'private'
    with pytest.raises(ValueError, match='REWARD_FACT_UNOBSERVABLE'):
        validate_reward_observability(source)


def test_temporal_diagnostics_follow_inherited_upstream_queries():
    source = computed_query_source()
    source['tool_implementations'].append({
        'tool_name': 'reduce_demand', 'operation': 'update', 'table': 'demand'})
    bind_reward_queries(source)
    findings = reward_temporal_diagnostics(source)
    assert len(findings) == 1
    assert findings[0]['table'] == 'demand'
    assert findings[0]['mutators'] == ['reduce_demand']
    assert '.where.capacity.eq.' in findings[0]['path']
    target = source['metric_implementations'][0]['expected']['targets'][0]
    target['expression'] = {'initial': target['expression']}
    assert reward_temporal_diagnostics(source) == []
    target['expression'] = {'at_tool': {'name': 'find_vehicle',
                                      'expression': target['expression']['initial']}}
    assert reward_temporal_diagnostics(source) == []


def test_temporal_diagnostics_include_external_events_and_ignore_literals():
    source = computed_query_source()
    source['business_lifecycle'] = {'dynamics': {'events': [
        {'id': 'competition', 'changes': [{'table': 'vehicles'}]}]}}
    bind_reward_queries(source)
    findings = reward_temporal_diagnostics(source)
    assert len(findings) == 1
    assert findings[0]['mutators'] == ['event:competition']
    target = source['metric_implementations'][0]['expected']['targets'][0]
    target['expression'] = {'literal': target['expression']}
    assert reward_temporal_diagnostics(source) == []
