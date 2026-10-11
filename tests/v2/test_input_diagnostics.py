import copy

from env_factory_v2.author_workbench import AuthorWorkbench
from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import simple, lit
from env_factory_v2.input_diagnostics import input_diagnostics


def first_match_source():
    source = simple()
    source['answers'][0].update(schema={'type': 'string'}, comparison='exact',
                               expected={'at': [source['answers'][0]['expected'], lit(0)]})
    source['scenarios'] = source['scenarios'][:2]
    source['scenarios'][0]['answer'] = {'ids': lit('I1')}
    source['scenarios'][1]['answer'] = {'ids': lit('incorrect')}
    source['probes'] = []
    return source


def test_empty_match_expression_crash_is_hint_not_new_acceptance_gate():
    source = first_match_source()
    original = copy.deepcopy(source)
    workbench = AuthorWorkbench(source=source)
    result = workbench.call('check_contract', {})
    assert result['passed'] is True
    diagnostic = result['input_diagnostics']
    assert diagnostic['variants_tested'] == 1
    assert [v['path'] for v in diagnostic['findings']] == ['/answers/0/expected']
    assert 'unresolved collection index' in diagnostic['findings'][0]['error']
    assert source == original and workbench.snapshot() == original


def test_guarded_empty_answer_has_no_crash_hint():
    source = first_match_source()
    previous = source['answers'][0]['expected']
    values = previous['at'][0]
    source['answers'][0]['expected'] = {'if': {
        'condition': {'gt': [{'count': [values]}, lit(0)]}, 'then': previous, 'else': lit('unavailable')}}
    assert input_diagnostics(compile_contract(source))['findings'] == []


def test_closed_input_domain_is_not_relaxed_to_manufacture_failure():
    source = first_match_source()
    source['tools'][0]['parameters']['properties']['size']['enum'] = ['M', 'L']
    diagnostic = input_diagnostics(compile_contract(source))
    assert diagnostic['variants_tested'] == 0
    assert diagnostic['findings'] == []


def test_empty_collection_is_not_itself_a_failure():
    diagnostic = input_diagnostics(compile_contract(simple()))
    assert diagnostic['variants_tested'] == 1
    assert diagnostic['findings'] == []


def test_diagnostics_respect_work_budget():
    diagnostic = input_diagnostics(compile_contract(first_match_source()), max_variants=0)
    assert diagnostic['variants_tested'] == 0 and diagnostic['findings'] == []
