from env_factory.generation.agent_authoring import state_predicate_diagnostic, unresolved_lookup_diagnostics
from env_factory.sandbox_runtime import ExpressionBusinessState


def test_missing_public_identity_explains_unsatisfied_preservation_target():
    baseline = {'events': [{'id': 'E1', 'code': 'different'}],
                'slots': [{'id': 'S1', 'event': 'E1', 'status': 'available'}]}
    state = ExpressionBusinessState(baseline, baseline)
    event = {'lookup': {'table': 'events', 'field': 'id', 'where': {'code': 'requested'}}}
    slot = {'lookup': {'table': 'slots', 'field': 'id', 'where': {'event': event}}}
    predicate = {'table': 'slots', 'where': {}, 'values': {'status': 'available'}, 'count': 1,
                 'value_expressions': {'id': {'initial': slot}}}
    diagnostic = state_predicate_diagnostic(predicate, state, baseline)
    assert diagnostic['expected_values'] is None and not diagnostic['satisfied']
    root = next(x for x in diagnostic['unresolved_lookups'] if x['table'] == 'events')
    assert root['matched_row_count'] == 0 and root['resolved_where'] == {'code': 'requested'}
    assert diagnostic['actual_rows'] == baseline['slots']


def test_unselected_branch_cannot_create_spurious_missing_checkpoint():
    state = ExpressionBusinessState({}, {}, {})
    expression = {'if': {'condition': {'literal': False},
        'then': {'at_tool': {'name': 'unreachable', 'expression': {'literal': 'unused'}}},
        'else': {'lookup': {'table': 'events', 'field': 'id', 'where': {'code': 'requested'}}}}}
    findings = unresolved_lookup_diagnostics(expression, state)
    assert len(findings) == 1 and findings[0]['reason'] == 'lookup_not_unique'
    assert '.if.else' in findings[0]['path']
