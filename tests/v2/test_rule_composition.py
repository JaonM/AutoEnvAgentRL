import pytest

from env_factory_v2.contract import compile_contract, validate_rule_graph
from env_factory_v2.examples import transaction, ref, lit
from env_factory_v2.expressions import ContractError, evaluate


def invoke(name, **arguments):
    return {'rule': {'id': name, 'arguments': arguments}}


def test_composed_rule_uses_current_values_and_exact_caller_arguments():
    rules = {'minimum': {'predicate': {'ge': [ref('row.balance'), ref('args.amount')]}},
             'available': {'predicate': invoke('minimum', amount=ref('args.requested'))}}
    validate_rule_graph(rules)
    context = {'row': {'balance': 20}, '_rules': rules}
    assert evaluate(invoke('available', requested=lit(12)), context) is True
    context['row']['balance'] = 10
    assert evaluate(invoke('available', requested=lit(12)), context) is False
    assert '_rule_stack' not in context


def test_composition_cannot_hide_transaction_row_fields():
    source = transaction()
    source['rules'].extend([
        {'id': 'leaf', 'public': 'Target status must be open.',
         'predicate': {'eq': [ref('row.status'), lit('open')]}},
        {'id': 'composed', 'public': 'Target status must be open.', 'predicate': invoke('leaf')}])
    source['tools'][1]['rule_ids'].append('composed')
    with pytest.raises(ContractError, match='row fields absent'):
        compile_contract(source)


@pytest.mark.parametrize('rules, message', [
    ({'A': {'predicate': invoke('A')}}, 'cycle'),
    ({'A': {'predicate': invoke('B')}, 'B': {'predicate': invoke('A')}}, 'cycle'),
    ({'A': {'predicate': invoke('missing')}}, 'unknown'),
    ({'A': {'predicate': invoke('B')}, 'B': {'predicate': {'eq': [ref('args.x'), lit(1)]}}}, 'argument mismatch'),
])
def test_invalid_composition_fails_compilation(rules, message):
    with pytest.raises(ContractError, match=message):
        validate_rule_graph(rules)


def test_depth_limit_and_runtime_cycle_guard():
    rules = {f'R{i}': {'predicate': invoke(f'R{i+1}')} for i in range(16)}
    rules['R16'] = {'predicate': lit(True)}
    with pytest.raises(ContractError, match='depth'):
        validate_rule_graph(rules)
    with pytest.raises(ContractError, match='cycle'):
        evaluate(invoke('A'), {'_rules': {'A': {'predicate': invoke('A')}}})
