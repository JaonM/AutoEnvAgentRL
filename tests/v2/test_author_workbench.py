import json
import pytest

from env_factory_v2.author_workbench import AuthorWorkbench, author_feedback
from env_factory_v2.contract import ContractError
from env_factory_v2.examples import direct, simple


@pytest.mark.parametrize('factory', [direct, simple])
def test_incremental_sections_compile_and_execute_real_recipes(factory):
    source = factory()
    bench = AuthorWorkbench()
    for section, value in source.items():
        assert bench.call('put_section', {'section': section, 'value_json': json.dumps(value)})['passed']
    assert bench.snapshot() == source
    result = bench.call('check_contract', {})
    assert result['passed'] and all(c['passed'] for c in result['checks'])
    assert result['scope'] == 'authored_recipe_validation_not_business_acceptance'


@pytest.mark.parametrize('value', ['{"a":1,"a":2}', '{broken', '[NaN]', '[Infinity]'])
def test_bad_section_leaves_previous_draft_unchanged(value):
    source = direct()
    bench = AuthorWorkbench(source=source)
    assert not bench.call('put_section', {'section': 'rules', 'value_json': value})['passed']
    assert bench.snapshot() == source


def test_local_edits_expose_actual_compiler_failure_and_can_repair_it():
    source = simple()
    bench = AuthorWorkbench(source=source)
    key = next(iter(bench.source['scenarios'][0]['answer']))
    assert bench.call('edit_contract', {'edits': [{'op': 'remove',
        'path': f'/scenarios/0/answer/{key}', 'value_json': 'null'}]})['passed']
    result = bench.call('check_contract', {})
    assert not result['passed'] and 'scenario answer fields mismatch' in result['error']
    assert bench.call('edit_contract', {'edits': [{'op': 'add',
        'path': f'/scenarios/0/answer/{key}',
        'value_json': json.dumps(source['scenarios'][0]['answer'][key])}]})['passed']
    assert bench.call('check_contract', {})['passed']


def test_unknown_tool_budget_and_snapshot_do_not_grant_other_access():
    bench = AuthorWorkbench(source=direct(), max_operations=1)
    snapshot = bench.snapshot()
    snapshot['task'] = 'changed'
    assert snapshot != bench.snapshot()
    assert not bench.call('shell', {'cmd': 'anything'})['passed']
    with pytest.raises(ContractError, match='budget exceeded'):
        bench.call('check_contract', {})


def test_compact_feedback_preserves_all_failure_evidence_and_case_counts():
    report = {'passed': False, 'checks': [
        {'id': 'good', 'kind': 'scenario', 'passed': True, 'error': None,
         'workflow': {'process': [{'long_passed_evidence': 'x'*1000}]}},
        {'id': 'bad', 'kind': 'probe', 'passed': False, 'error': None,
         'failure_evidence': {'answer': {'actual': 7}}, 'workflow': {'passed': False}}]}
    compact = author_feedback(report)
    assert compact['checks'][1] == report['checks'][1]
    assert compact['checks'][0] == {'id': 'good', 'kind': 'scenario', 'passed': True, 'error': None}
    assert compact['check_count'] == 2 and compact['failed_count'] == 1
    assert len(json.dumps(compact)) < len(json.dumps(report))
    compact['checks'][1]['failure_evidence']['answer']['actual'] = 9
    assert report['checks'][1]['failure_evidence']['answer']['actual'] == 7
