import pytest

from env_factory_v2.contract import validate_rule_links, free_row_fields
from env_factory_v2.expressions import ContractError, evaluate, validate_expr


def test_outer_project_key_survives_inner_expense_rows_and_reordering():
    expr = {'sum': [{'map': {'items': {'ref': 'data.projects'}, 'value': {'let': {
        'bindings': {'project_id': {'ref': 'row.id'}},
        'value': {'sum': [{'map': {'items': {'filter': {
            'items': {'ref': 'data.expenses'}, 'value': {'rule': {
                'id': 'membership', 'arguments': {'project_id': {'ref': 'vars.project_id'}}}}}},
            'value': {'ref': 'row.amount'}}}]}}}}}]}
    rules = {'membership': {'predicate': {'eq': [{'ref': 'row.project_id'}, {'ref': 'args.project_id'}]}}}
    ctx = {'data': {'projects': [{'id': 'P1'}, {'id': 'P2'}], 'expenses': [
        {'id': 'X1', 'project_id': 'P1', 'amount': 7},
        {'id': 'X2', 'project_id': 'P2', 'amount': 11},
        {'id': 'X3', 'project_id': 'P1', 'amount': 3},
        {'id': 'P1', 'project_id': 'P3', 'amount': 999}]}, '_rules': rules}
    validate_expr(expr, roots={'data'})
    validate_rule_links(expr, rules, shared_filters=True)
    assert evaluate(expr, ctx) == 21
    ctx['data']['projects'].reverse()
    ctx['data']['expenses'].reverse()
    assert evaluate(expr, ctx) == 21
    assert 'vars' not in ctx


def test_bindings_are_simultaneous_lexical_and_do_not_mutate_context():
    ctx = {'vars': {'a': 9}, 'row': {'id': 'P1'}}
    expr = {'let': {'bindings': {'a': {'literal': 2}, 'b': {'ref': 'vars.a'}},
                    'value': {'list': [{'ref': 'vars.a'}, {'ref': 'vars.b'}]}}}
    validate_expr(expr, roots={'vars'})
    assert evaluate(expr, ctx) == [2, 9]
    assert ctx == {'vars': {'a': 9}, 'row': {'id': 'P1'}}
    assert free_row_fields({'let': {'bindings': {'key': {'ref': 'row.id'}},
                                  'value': {'ref': 'vars.key'}}}) == {'id'}


@pytest.mark.parametrize('bindings', [{}, {'bad.name': {'literal': 1}},
                                    {'key': {'ref': 'evidence.secret'}}])
def test_bindings_cannot_bypass_names_or_reference_boundaries(bindings):
    with pytest.raises(ContractError):
        validate_expr({'let': {'bindings': bindings, 'value': {'ref': 'vars.key'}}}, roots={'data'})


def test_local_variables_are_not_available_outside_binding_scope():
    with pytest.raises(ContractError, match='forbidden'):
        validate_expr({'ref': 'vars.key'}, roots={'data'})


def test_record_row_cannot_be_used_as_outer_collection_index():
    tables = {'items': {'schema': {'properties': {'id': {'type': 'string'}}}}}
    wrong = {'filter': {'items': {'ref': 'initial.items'},
                        'value': {'eq': [{'ref': 'row.id'},
                                        {'at': [{'ref': 'user.ids'}, {'ref': 'row'}]}]}}}
    with pytest.raises(ContractError, match='collection index is a record from table items'):
        validate_rule_links(wrong, {}, tables=tables)
    correct = {'map': {'items': {'range': {'start': {'literal': 0},
                                         'stop': {'literal': 2}, 'max_items': 2}},
                       'value': {'let': {'bindings': {'index': {'ref': 'row'}},
                                        'value': {'filter': {
                                            'items': {'ref': 'initial.items'},
                                            'value': {'eq': [{'ref': 'row.id'}, {'at': [
                                                {'ref': 'user.ids'}, {'ref': 'vars.index'}]}]}}}}}}}
    validate_expr(correct, roots={'initial', 'user'})
    validate_rule_links(correct, {}, tables=tables)
    assert evaluate(correct, {'initial': {'items': [{'id': 'B'}, {'id': 'A'}]},
                              'user': {'ids': ['A', 'B']}}) == [[{'id': 'A'}], [{'id': 'B'}]]


def test_lexical_capture_around_public_filter_preserves_rule_and_boundary_semantics():
    rules = {'eligible': {'predicate': {'and': [
        {'eq': [{'ref': 'row.project_id'}, {'ref': 'args.project_id'}]},
        {'gt': [{'ref': 'row.amount'}, {'literal': 10}]}]}}}
    expr = {'filter': {'items': {'ref': 'data.expenses'}, 'value': {'let': {
        'bindings': {'key': {'ref': 'row.project_id'}}, 'value': {'rule': {
            'id': 'eligible', 'arguments': {'project_id': {'ref': 'vars.key'}}}}}}}}
    validate_expr(expr, roots={'data'})
    validate_rule_links(expr, rules, shared_filters=True)
    rows = [{'project_id': 'P1', 'amount': 10}, {'project_id': 'P2', 'amount': 11}]
    assert evaluate(expr, {'data': {'expenses': rows}, '_rules': rules}) == [rows[1]]


@pytest.mark.parametrize('predicate', [
    {'literal': True},
    {'eq': [{'ref': 'row.amount'}, {'literal': 10}]},
    {'or': [{'rule': {'id': 'eligible', 'arguments': {}}}, {'literal': True}]},
])
def test_lexical_wrapper_cannot_hide_unshared_eligibility(predicate):
    expr = {'filter': {'items': {'ref': 'data.expenses'}, 'value': {'let': {
        'bindings': {'key': {'ref': 'row.amount'}}, 'value': predicate}}}}
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate_rule_links(expr, {'eligible': {'predicate': {'literal': False}}}, shared_filters=True)
