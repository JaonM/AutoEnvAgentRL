import copy

from env_factory.sandbox_runtime import BusinessGoalEvaluator as Goal


def test_untracked_fee_identifies_exact_row_field_and_goal_without_changing_verdict():
    baseline = {'reservations': [{'id': 'R1', 'status': 'draft', 'fee': 0}]}
    state = {'reservations': [{'id': 'R1', 'status': 'confirmed', 'fee': 900}]}
    goal = {'table_primary_keys': {'reservations': ['id']}, 'row_predicates': [
        {'table': 'reservations', 'where': {'id': 'R1'}, 'values': {'status': 'confirmed'}, 'count': 1}]}
    evidence = []
    assert not Goal.preserves_unrelated(goal, baseline, state, diagnostics=evidence)
    assert not Goal.preserves_unrelated(goal, baseline, state)
    issue = evidence[0]
    assert issue['identity'] == {'id': 'R1'}
    assert issue['unexpected_fields'] == ['fee']
    assert issue['candidate_goals'][0]['path'] == 'goal_contract.row_predicates[0]'
    assert next(change for change in issue['changes'] if change['field'] == 'fee') == {
        'field': 'fee', 'before': 0, 'after': 900, 'before_present': True, 'after_present': True}
    corrected = copy.deepcopy(goal)
    corrected['row_predicates'][0]['values']['fee'] = 900
    evidence = []
    assert Goal.preserves_unrelated(corrected, baseline, state, diagnostics=evidence)
    assert evidence == []


def test_wrong_declared_value_is_distinguished_from_undeclared_field():
    baseline = {'rows': [{'id': 1, 'quantity': 10}]}
    state = {'rows': [{'id': 1, 'quantity': 7}]}
    goal = {'table_primary_keys': {'rows': ['id']}, 'row_predicates': [
        {'table': 'rows', 'where': {'id': 1}, 'values': {'quantity': 6}, 'count': 1}]}
    evidence = []
    assert not Goal.preserves_unrelated(goal, baseline, state, diagnostics=evidence)
    assert evidence[0]['unexpected_fields'] == []
    assert evidence[0]['candidate_goals'][0]['expected_values'] == {'quantity': 6}
    assert evidence[0]['changes'][0]['after'] == 7


def test_external_updates_are_excluded_from_preservation_diagnostics():
    baseline = {'rows': [{'id': 1, 'quantity': 10}], 'other': [{'id': 2, 'quantity': 20}]}
    state = {'rows': [{'id': 1, 'quantity': 6}], 'other': [{'id': 2, 'quantity': 18}]}
    events = [{'event': 'external_business_change', 'payload': {
        'table': 'other', 'selector': {'id': 2}, 'values': {'quantity': 18}},
        'result': {'updated_count': 1}}]
    goal = {'table_primary_keys': {'rows': ['id']}, 'row_predicates': [
        {'table': 'rows', 'where': {'id': 1}, 'values': {'quantity': 6}, 'count': 1}]}
    evidence = []
    adjusted = Goal.external_baseline(baseline, events)
    assert Goal.preserves_unrelated(goal, adjusted, state, diagnostics=evidence)
    assert evidence == []
    assert baseline['other'][0]['quantity'] == 20
