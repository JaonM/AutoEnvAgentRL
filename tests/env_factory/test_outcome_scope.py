import copy

from env_factory.production.world import task_schema, outcome_problems


def test_only_explicit_public_selection_is_bound_to_each_goal_not_private_facts():
    process = {'kind': 'inquiry', 'params': {'fields': 'public field names', 'member_id': 'private ID'},
               'outcome_scope_params': ['fields'],
               'outcome': {'type': 'object', 'properties': {'result': {'type': 'string'}}}}
    schema = {'lookup': process}
    goals = [{'process': 'lookup', 'params': {'fields': ['benefit'], 'member_id': 'SECRET_A'}},
             {'process': 'lookup', 'params': {'fields': ['expiry'], 'member_id': 'SECRET_B'}}]
    original = copy.deepcopy(schema)
    result = task_schema(schema, goals)['fields']
    assert 'benefit' in result['goal_1']['description'] and 'expiry' not in result['goal_1']['description']
    assert 'expiry' in result['goal_2']['description'] and 'benefit' not in result['goal_2']['description']
    assert 'SECRET' not in str(result)
    assert schema == original
    process.pop('outcome_scope_params')
    assert 'Requested outcome scope' not in str(task_schema(schema, goals))


def test_unknown_scope_parameter_is_rejected_before_authoring():
    schema = {'p': {'kind': 'inquiry', 'params': {'fields': 'public selection'},
                    'outcome_scope_params': ['not_a_param'],
                    'outcome': {'type': 'object', 'properties': {'result': {'type': 'string'}}}}}
    assert any('outcome_scope_params' in p for p in outcome_problems(schema))
