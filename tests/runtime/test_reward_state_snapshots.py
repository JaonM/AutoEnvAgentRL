"""Decision-time rewards remain valid across events, writes and transport retry."""
import copy
import json

import pytest

from env_factory.sandbox_runtime import (
    BusinessLifecycle, ContractToolRegistry, DeclarativeMetricEvaluator as E,
    DeclarativeToolCompiler, EpisodeStore, ExpressionBusinessState,
    ManifestDataStore, RewardStateSnapshots, SandboxError,
)


LOOKUP = {"lookup": {"table": "inventory", "field": "available", "where": {"id": 1}}}
CHECKPOINT = {"at_tool": {"name": "read_stock", "expression": LOOKUP}}


def runtime(tmp_path, *, fault=None):
    schema = {"columns": [{"name": "id", "type": "INTEGER", "nullable": False},
                          {"name": "available", "type": "INTEGER", "nullable": False}],
              "primary_key": ["id"]}
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "rows.jsonl").write_text('{"id":1,"available":20}\n')
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    data = ManifestDataStore({"environment_mode": "stateful", "tables": [
        {"table_name": "inventory", "schema_file": "schema.json", "rows_file": "rows.jsonl"}]},
        tmp_path, store)
    data.reset()
    implementations = [
        {"tool_name": "read_stock", "operation": "select", "table": "inventory"},
        {"tool_name": "reserve", "operation": "update", "table": "inventory",
         "selector": {"id": "id"}, "changes": {"available": "available"}},
    ]
    tools = [{"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": props, "required": list(props), "additionalProperties": False}}}
        for name, props in [("read_stock", {}), ("reserve", {
            "id": {"type": "integer"}, "available": {"type": "integer"}})]]
    dynamics = {"events": [{"id": "competition", "at_tick": 1, "changes": [
        {"table": "inventory", "selector": {"id": 1}, "values": {"available": 16}}]}]}
    if fault:
        dynamics["faults"] = [{"id": "read_failure", "tool": "read_stock",
                               "call_index": 1, "phase": fault, "status": 504}]
    lifecycle = BusinessLifecycle({"version": "1.0", "dynamics": dynamics}, data, store,
                                  [item["tool_name"] for item in implementations])
    registry = ContractToolRegistry(tools, DeclarativeToolCompiler(data).compile_all(implementations),
        event_recorder=store.event, lifecycle=lifecycle,
        reward_snapshots=RewardStateSnapshots(data, implementations, [CHECKPOINT]))
    return store, data, registry


def context(store, data):
    return ExpressionBusinessState(data._all(), data.baseline,
                                   store.get_state(RewardStateSnapshots.KEY, {}))


def test_result_checkpoint_ignores_unrelated_calls_before_bound_recovery(tmp_path):
    store, data, registry = runtime(tmp_path)
    expression = {'tool_result': {'name': 'read_stock', 'path': '$.records[0].available'}}
    binding = {'source': 'trajectory', 'operator': 'contains_tool_call', 'expected': {
        'tool_name': 'read_stock', 'arguments': {'region': 'east'}, 'by_user_script': {
            'script-1': [{'tool_name': 'read_stock', 'arguments': {'region': 'east'}}],
            'script-2': [{'tool_name': 'read_stock', 'arguments': {'region': 'west'}}]}}}
    snapshots = RewardStateSnapshots(data, registry.reward_snapshots.implementations, [expression, binding])
    store.event('user_protocol', {'script_id': 'script-1'}, {})
    snapshots.capture('read_stock', {'records': [{'available': 99}]}, {'region': 'west'})
    assert E._value_expression(expression, context(store, data)) is None
    snapshots.capture('read_stock', {'records': [{'available': 20}]}, {'region': 'east'})
    assert E._value_expression(expression, context(store, data)) == 20
    snapshots.capture('read_stock', {'records': [{'available': 88}]}, {'region': 'east'})
    assert E._value_expression(expression, context(store, data)) == 20


def test_checkpoint_tracks_observed_stock_not_initial_or_final(tmp_path):
    store, data, registry = runtime(tmp_path)
    assert E._value_expression(CHECKPOINT, context(store, data)) is None
    registry.execute("read_stock", {})
    registry.execute("reserve", {"id": 1, "available": 4})
    registry.execute("read_stock", {})  # a later query must not rewrite the decision
    state = context(store, data)
    assert E._value_expression({"initial": LOOKUP}, state) == 20
    assert E._value_expression(CHECKPOINT, state) == 16
    assert E._value_expression(LOOKUP, state) == 4
    targets = {"answer_format": "json_object", "targets": [{"key": "available", "expression": CHECKPOINT}]}
    # Both typed answer and numeric expression evaluation accept checkpoint values.
    assert E._numeric_expression(CHECKPOINT, state) == 16
    assert E._value_targets_match('{"available":16}', targets, state)
    assert not E._value_targets_match('{"available":20}', targets, state)
    events = [event for event in store.replay()["events"] if event["event"] == "reward_state_checkpoint"]
    assert len(events) == 1
    assert "inventory" not in events[0]["result"]  # private data is not exported in replay


@pytest.mark.parametrize("phase", ["before", "after_commit"])
def test_snapshot_fault_and_idempotency_semantics(tmp_path, phase):
    store, data, registry = runtime(tmp_path, fault=phase)
    with pytest.raises(SandboxError):
        registry.execute("read_stock", {}, idem_key="same-read")
    snapshots = store.get_state(RewardStateSnapshots.KEY, {})
    assert ("read_stock" in snapshots) == (phase == "after_commit")
    registry.execute("read_stock", {}, idem_key="same-read")
    assert E._value_expression(CHECKPOINT, context(store, data)) == 16
    assert len([event for event in store.replay()["events"]
                if event["event"] == "reward_state_checkpoint"]) == 1


def test_snapshot_persists_and_isolates_episodes(tmp_path):
    store, data, registry = runtime(tmp_path)
    episode = store.current()
    registry.execute("read_stock", {})
    saved = copy.deepcopy(store.get_state(RewardStateSnapshots.KEY))
    reopened = EpisodeStore(tmp_path / "episodes.sqlite3")
    with reopened.episode_context(episode.episode_id):
        assert reopened.get_state(RewardStateSnapshots.KEY) == saved
    store.reset(seed=9)
    data.reset()
    assert E._value_expression(CHECKPOINT, context(store, data)) is None
    with store.episode_context(episode.episode_id):
        assert store.get_state(RewardStateSnapshots.KEY) == saved


def test_write_or_unknown_checkpoint_is_rejected():
    for implementations in ([], [{"tool_name": "read_stock", "operation": "update"}]):
        with pytest.raises(SandboxError, match="read-only"):
            RewardStateSnapshots.validate_tools(implementations, CHECKPOINT)
    with pytest.raises(SandboxError, match="requires name and expression"):
        E._value_expression({"at_tool": {"name": "read_stock"}}, {})
    # Missing checkpoint still validates its inner expression.
    with pytest.raises(SandboxError):
        E._value_expression({"at_tool": {"name": "read_stock", "expression": {"unknown": 1}}}, {})


def test_invalid_result_does_not_establish_checkpoint_and_reset_clears_it(tmp_path):
    store, data, registry = runtime(tmp_path)
    registry.output_schemas['read_stock'] = {'type': 'object', 'required': ['missing']}
    with pytest.raises(SandboxError, match='tool_result'):
        registry.execute('read_stock', {})
    assert store.get_state(RewardStateSnapshots.KEY, {}) == {}
    registry.output_schemas.clear()
    registry.execute('read_stock', {})
    assert E._value_expression(CHECKPOINT, context(store, data)) == 16
    # Resetting the same episode ID must also clear checkpoints.
    store.reset(episode_id=store.current().episode_id, seed=17)
    data.reset()
    assert E._value_expression(CHECKPOINT, context(store, data)) is None


def test_state_target_uses_checkpoint_and_requires_real_write(tmp_path):
    from env_factory.sandbox_runtime import ContractRewardGate
    store, data, registry = runtime(tmp_path)
    predicate = {"table": "inventory", "where": {"id": 1}, "values": {}, "count": 1,
                 "value_expressions": {"available": {"op": "sub", "args": [CHECKPOINT, {"literal": 12}]}}}
    rule = {"metric_id": "reserved", "source": "business_state", "path": "$",
            "operator": "state_predicates", "expected": [predicate]}
    metric = {"id": "reserved", "category": "outcome", "score_range": [0, 1]}
    gate = ContractRewardGate({"training_contract": {"category": "simple_agentic"},
        "capability_dag": {"nodes": ["read_stock", "reserve"]},
        "goal_contract": {"row_predicates": [predicate], "requires_state_change": True,
                          "table_primary_keys": {"inventory": ["id"]}}}, [metric])
    def score():
        ctx = {"business_state": data._all(), "initial_business_state": data.baseline,
               "reward_state_snapshots": store.get_state(RewardStateSnapshots.KEY, {}),
               "trajectory": store.replay()}
        return gate.apply({"reserved": E().evaluate(rule, ctx)}, ctx)["reserved"]
    registry.execute("read_stock", {})
    assert score() == 0
    registry.execute("reserve", {"id": 1, "available": 4})
    assert score() == 1
    registry.execute("reserve", {"id": 1, "available": 8})  # incorrect use of reset stock
    assert score() == 0


def test_selected_record_binding_uses_actual_order_and_first_observation(tmp_path):
    from env_factory.sandbox_runtime import RewardStateSnapshots, DeclarativeMetricEvaluator, ExpressionBusinessState
    from tests.runtime.test_nested_transactions import business_data
    store, data = business_data(tmp_path)
    expression = {'tool_result': {'name': 'search', 'path': '$.records[0].id'}}
    snapshots = RewardStateSnapshots(data, [{'tool_name': 'search', 'operation': 'select'}], [expression])
    state = lambda: ExpressionBusinessState(data._all(), data.baseline, store.get_state(snapshots.KEY, {}))
    assert DeclarativeMetricEvaluator._value_expression(expression, state()) is None
    snapshots.capture('search', {'records': [{'id': 2}, {'id': 1}]})
    assert DeclarativeMetricEvaluator._value_expression(expression, state()) == 2
    snapshots.capture('search', {'records': [{'id': 1}]})
    assert DeclarativeMetricEvaluator._value_expression(expression, state()) == 2


@pytest.mark.parametrize('rows', [[], [{'id': 1, 'available': 16}],
    [{'id': i, 'available': i * 2} for i in range(64)]])
def test_structured_observed_result_tracks_full_typed_collection(tmp_path, rows):
    store, data, registry = runtime(tmp_path)
    expression = {'tool_result': {'name': 'read_stock', 'path': '$.records'}}
    snapshots = RewardStateSnapshots(data, registry.reward_snapshots.implementations, [expression])
    assert E._value_expression(expression, context(store, data)) is None
    snapshots.capture('read_stock', {'records': rows}, {})
    actual = E._value_expression(expression, context(store, data))
    assert actual == rows
    assert E._json_value_equal(actual, rows)
    if rows:
        wrong = copy.deepcopy(rows); wrong[0]['available'] = str(wrong[0]['available'])
        assert not E._json_value_equal(actual, wrong)
        assert not E._json_value_equal(actual, rows[:-1])
        actual[0]['available'] = -100
        assert E._value_expression(expression, context(store, data)) == rows


@pytest.mark.parametrize('value', [[float('inf')],
    [float('nan')], list(range(65)), {str(i):i for i in range(65)}])
def test_structured_result_rejects_nonfinite_and_oversized_values(value):
    expression = {'tool_result': {'name': 'read', 'path': '$.records'}}
    state = ExpressionBusinessState({}, snapshots={'$tool_results': {'read': {'records': value}}})
    assert E.value_resolution(expression, state) == {'resolved': False, 'value': None}


@pytest.mark.parametrize('value', [None, [None], [{'value': None}]])
def test_structured_null_is_observed_evidence_but_missing_records_are_not(value):
    expression = {'tool_result': {'name': 'read', 'path': '$.records'}}
    state = ExpressionBusinessState({}, snapshots={'$tool_results': {'read': {'records': value}}})
    assert E.value_resolution(expression, state) == {'resolved': True, 'value': value}
    missing = ExpressionBusinessState({}, snapshots={'$tool_results': {'read': {}}})
    assert E.value_resolution(expression, missing) == {'resolved': False, 'value': None}


def test_structured_result_does_not_change_script_binding_or_checkpoint_time(tmp_path):
    store, data, registry = runtime(tmp_path)
    expression = {'tool_result': {'name':'read_stock','path':'$.records'}}
    binding = {'source':'trajectory','operator':'contains_tool_call','expected': {
        'tool_name':'read_stock','arguments':{'region':'east'},'by_user_script': {
            'script-1':[{'tool_name':'read_stock','arguments':{'region':'east'}}],
            'script-2':[{'tool_name':'read_stock','arguments':{'region':'west'}}]}}}
    snapshots = RewardStateSnapshots(data, registry.reward_snapshots.implementations, [expression,binding])
    store.event('user_protocol',{'script_id':'script-2'}, {})
    snapshots.capture('read_stock', {'records':[{'id':1}]}, {'region':'east'})
    assert E._value_expression(expression, context(store,data)) is None
    snapshots.capture('read_stock', {'records':[]}, {'region':'west'})
    snapshots.capture('read_stock', {'records':[{'id':2}]}, {'region':'west'})
    assert E._value_expression(expression, context(store,data)) == []


def test_structured_tool_value_rejects_deep_nesting():
    value = 'leaf'
    for _ in range(14): value = [value]
    state = ExpressionBusinessState({}, snapshots={'$tool_results': {'read': {'records':value}}})
    assert E._value_expression({'tool_result':{'name':'read','path':'$.records'}},state) is None


def test_query_argument_survives_empty_result_and_preserves_active_script(tmp_path):
    store, data, registry = runtime(tmp_path)
    expression={'tool_argument':{'name':'read_stock','path':'$.region'}}
    binding={'source':'trajectory','operator':'contains_tool_call','expected':{
        'tool_name':'read_stock','arguments':{'region':'east'},'by_user_script':{
            'script-1':[{'tool_name':'read_stock','arguments':{'region':'east'}}],
            'script-2':[{'tool_name':'read_stock','arguments':{'region':'west'}}]}}}
    snapshots=RewardStateSnapshots(data,registry.reward_snapshots.implementations,[expression,binding])
    store.event('user_protocol',{'script_id':'script-2'}, {})
    snapshots.capture('read_stock',{'records':[]},{'region':'east'})
    assert E._value_expression(expression,context(store,data)) is None
    snapshots.capture('read_stock',{'records':[]},{'region':'west'})
    assert E._value_expression(expression,context(store,data))=='west'
    snapshots.capture('read_stock',{'records':[{'available':7}]},{'region':'east'})
    assert E._value_expression(expression,context(store,data))=='west'


def test_arguments_require_declared_business_tool_and_valid_reference():
    expression={'tool_argument':{'name':'write','path':'$.quantity'}}
    assert RewardStateSnapshots.validate_tools(
        [{'tool_name':'write','operation':'update'}],expression) == {'write'}
    for implementations in ([], [{'tool_name':'write','operation':'unknown'}]):
        with pytest.raises(SandboxError,match='declared business tools'):
            RewardStateSnapshots.validate_tools(implementations,expression)
    with pytest.raises(SandboxError,match='tool_argument'):
        RewardStateSnapshots.requested({'tool_argument':{'name':'read','path':''}})
    state=ExpressionBusinessState({},snapshots={'$tool_arguments':{'read':{'quantity':7}}})
    assert E._numeric_expression({'tool_argument':{'name':'read','path':'$.quantity'}},state)==7


def write_runtime(tmp_path, operation, *, fault=None):
    from tests.runtime.test_nested_transactions import business_data
    store, data = business_data(tmp_path)
    implementation = {'tool_name': 'write', 'operation': operation, 'table': 'inventory',
                      'selector': {'id': 'id'}, 'changes': {'quantity': 'quantity'},
                      'values': {'id': 'id', 'quantity': 'quantity'}}
    schema = {'type': 'object', 'properties': {'id': {'type': 'integer'},
              'quantity': {'type': 'integer'}}, 'required': ['id', 'quantity'],
              'additionalProperties': False}
    expressions = [{'tool_result': {'name': 'write', 'path': '$'}},
                   {'tool_argument': {'name': 'write', 'path': '$'}}]
    config = {'version': '1.0'}
    if fault:
        config['dynamics'] = {'faults': [{'id': 'write_failure', 'tool': 'write',
            'call_index': 1, 'phase': fault, 'status': 504}]}
    registry = ContractToolRegistry(
        [{'type': 'function', 'function': {'name': 'write', 'parameters': schema}}],
        DeclarativeToolCompiler(data).compile_all([implementation]),
        event_recorder=store.event,
        lifecycle=BusinessLifecycle(config, data, store, ['write']),
        reward_snapshots=RewardStateSnapshots(data, [implementation], expressions))
    return store, data, registry, expressions


@pytest.mark.parametrize('operation', ['insert', 'update', 'delete'])
def test_committed_write_observations_are_first_actual_result_not_state_checkpoint(tmp_path, operation):
    store, data, registry, (result_ref, args_ref) = write_runtime(tmp_path, operation)
    args = {'id': 2 if operation == 'insert' else 1, 'quantity': 4}
    assert E.value_resolution(result_ref, context(store, data))['resolved'] is False
    actual = registry.execute('write', args, idem_key='write-once')
    expected = {'insert': {'record': args, 'count': 1},
                'update': {'updated_count': 1}, 'delete': {'deleted_count': 1}}[operation]
    assert actual == expected
    assert E.value_resolution(result_ref, context(store, data)) == {'resolved': True, 'value': expected}
    assert E._value_expression(args_ref, context(store, data)) == args
    assert 'write' not in store.get_state(RewardStateSnapshots.KEY)
    assert registry.execute('write', args, idem_key='write-once') == expected
    registry.execute('write', {'id': 3, 'quantity': 99})
    assert E._value_expression(result_ref, context(store, data)) == expected
    assert E._value_expression(args_ref, context(store, data)) == args
    assert len([e for e in store.replay()['events'] if e['event'] == 'reward_tool_observation']) == 1
    with pytest.raises(SandboxError, match='read-only'):
        RewardStateSnapshots.validate_tools(registry.reward_snapshots.implementations,
            {'at_tool': {'name': 'write', 'expression': {'literal': 1}}})


@pytest.mark.parametrize('operation', ['insert', 'update', 'delete'])
@pytest.mark.parametrize('phase', ['before', 'after_commit'])
def test_write_observation_transport_fault_preserves_only_committed_evidence(tmp_path, operation, phase):
    store, data, registry, (result_ref, args_ref) = write_runtime(tmp_path, operation, fault=phase)
    args = {'id': 2 if operation == 'insert' else 1, 'quantity': 4}
    before = copy.deepcopy(data._all())
    with pytest.raises(SandboxError):
        registry.execute('write', args, idem_key='retry')
    assert E.value_resolution(result_ref, context(store, data))['resolved'] == (phase == 'after_commit')
    assert E.value_resolution(args_ref, context(store, data))['resolved'] == (phase == 'after_commit')
    assert (data._all() != before) == (phase == 'after_commit')
    actual = registry.execute('write', args, idem_key='retry')
    assert E._value_expression(result_ref, context(store, data)) == actual
    assert len([e for e in store.replay()['events'] if e['event'] == 'reward_tool_observation']) == 1


@pytest.mark.parametrize('operation', ['insert', 'update', 'delete'])
@pytest.mark.parametrize('failure', ['invalid_result', 'outer_rollback'])
def test_write_and_observation_roll_back_together(tmp_path, operation, failure):
    store, data, registry, expressions = write_runtime(tmp_path, operation)
    args = {'id': 2 if operation == 'insert' else 1, 'quantity': 4}
    before = copy.deepcopy(data._all())
    if failure == 'invalid_result':
        registry.output_schemas['write'] = {'type': 'object', 'required': ['missing']}
        with pytest.raises(SandboxError, match='tool_result'):
            registry.execute('write', args, idem_key='retry')
        registry.output_schemas.clear()
    else:
        with pytest.raises(RuntimeError, match='rollback'):
            with store.transaction():
                registry.execute('write', args, idem_key='retry')
                assert E.value_resolution(expressions[0], context(store, data))['resolved']
                raise RuntimeError('rollback')
    assert data._all() == before
    for expression in expressions:
        assert E.value_resolution(expression, context(store, data))['resolved'] is False
    assert not [e for e in store.replay()['events'] if e['event'] == 'reward_tool_observation']
    actual = registry.execute('write', args, idem_key='retry')
    assert E._value_expression(expressions[0], context(store, data)) == actual
