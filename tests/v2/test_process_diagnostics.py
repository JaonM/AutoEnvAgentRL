import copy
from unittest.mock import patch

import pytest

from env_factory_v2.authoring.author_workbench import AuthorWorkbench
from env_factory_v2.contracts.contract import compile_contract, ContractError
from env_factory_v2.examples import simple
from env_factory_v2.validation.process_diagnostics import process_diagnostics
from env_factory_v2.validation.verification import run_scenario


def extra_question():
    source = compile_contract(simple())['source']
    source['time']['seconds_per_action'] = 0
    source['user']['facts']['unused_reference'] = 'private-reference'
    source['user']['stages'].append({'id': 'reference', 'kind': 'fact',
                                   'fact': 'unused_reference', 'depends_on': []})
    source['success'][0]['required_stages'].append('reference')
    source['scenarios'][0]['steps'].insert(0, {'kind': 'user', 'stage': 'reference',
                                           'question': 'What is the reference?'})
    return source


def test_same_world_answer_and_safety_can_be_rejected_by_extra_question():
    compiled = compile_contract(extra_question())
    before = copy.deepcopy(compiled)
    result = process_diagnostics(compiled)
    assert compiled == before
    assert result['findings'] and all(h['omitted_fact_stage'] == 'reference' for h in result['findings'])
    hint = next(h for h in result['findings'] if h['scenario'] == 'correct')
    assert hint['omitted_fact_stage'] == 'reference'
    assert hint['rejected_branches'] == ['complete']
    assert hint['same_answer_and_world'] and hint['safety_and_goals_passed']
    assert not hint['semantic_communication_checked']
    variant = copy.deepcopy(compiled['source']['scenarios'][0])
    variant['steps'] = hint['steps']
    assert not run_scenario(compiled, variant)['outcome']['success']  # No gate was waived.


def test_author_hint_does_not_fail_previously_valid_local_check():
    result = AuthorWorkbench(source=extra_question()).call('check_contract', {})
    assert result['passed'] and result['process_diagnostics']['findings']
    assert 'not_business_verdict' in result['process_diagnostics']['scope']


@pytest.mark.parametrize('change', ['no_gate', 'answer_dependency', 'downstream_dependency', 'time_dependency'])
def test_no_hint_when_question_is_not_the_isolated_observable_gate(change):
    source = extra_question()
    if change == 'no_gate':
        source['success'][0]['required_stages'].remove('reference')
    elif change == 'answer_dependency':
        source['answers'][0]['expected'] = {'if': {
            'condition': {'eq': [{'ref': 'user.unused_reference'}, {'literal': 'private-reference'}]},
            'then': source['answers'][0]['expected'], 'else': {'literal': []}}}
    elif change == 'downstream_dependency':
        source['user']['stages'][0]['depends_on'] = ['reference']
        source['user']['stages'].reverse()
    else:
        source['time']['seconds_per_action'] = 1
    assert not process_diagnostics(compile_contract(source))['findings']


def test_zero_budget_executes_no_additional_trajectories():
    with patch('env_factory_v2.validation.process_diagnostics.run_scenario', side_effect=AssertionError('no execution')):
        result = process_diagnostics(compile_contract(extra_question()), max_variants=0)
    assert result['variants_tested'] == result['baselines_tested'] == 0


def test_bounded_variants_and_required_query_fact_preservation():
    source = extra_question()
    # A real required fact is used in the query's named rule, so it is not a candidate.
    result = process_diagnostics(compile_contract(source), max_variants=1)
    assert result['variants_tested'] <= 1
    assert all(h['omitted_fact_stage'] == 'reference' for h in result['findings'])
    with pytest.raises(ContractError, match='budget'):
        process_diagnostics(compile_contract(source), max_variants=17)


def test_diagnostics_never_remove_authorization_or_execute_write_paths():
    from env_factory_v2.examples import transaction
    compiled = compile_contract(transaction())
    original = run_scenario
    def guarded(compiled, scenario, **kwargs):
        transactions = {t['id'] for t in compiled['source']['tools'] if t['kind'] == 'transaction'}
        assert not any(s.get('tool') in transactions for s in scenario['steps'])
        return original(compiled, scenario, **kwargs)
    with patch('env_factory_v2.validation.process_diagnostics.run_scenario', side_effect=guarded):
        result = process_diagnostics(compiled)
    fact_stages = {s['id'] for s in compiled['source']['user']['stages'] if s['kind'] == 'fact'}
    assert all(h['omitted_fact_stage'] in fact_stages for h in result['findings'])
