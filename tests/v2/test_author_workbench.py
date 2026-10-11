import json
import pytest

from env_factory_v2.author_workbench import AuthorWorkbench, author_feedback
from env_factory_v2.contract import ContractError
from env_factory_v2.examples import direct, simple


@pytest.mark.parametrize('factory', [direct, simple])
def test_incremental_sections_compile_and_execute_real_recipes(factory):
    source = factory()
    from test_business_definition import split
    environment, task = split(source)
    definition = {'environment': environment, 'task': task}
    bench = AuthorWorkbench()
    for section, value in definition.items():
        assert bench.call('put_section', {'section': section, 'value_json': json.dumps(value)})['passed']
    assert bench.snapshot() == definition
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


def test_read_existing_draft_then_edit_without_rewriting_other_sections():
    source = simple()
    bench = AuthorWorkbench(source=source)
    response = bench.call('read_contract', {'path': '/scenarios/0/answer', 'offset': 0})
    assert response['complete'] and response['next_offset'] is None
    assert json.loads(response['json_fragment']) == source['scenarios'][0]['answer']
    assert bench.snapshot() == source
    assert bench.call('edit_contract', {'edits': [{'op': 'replace', 'path': '/task',
                         'value_json': json.dumps('New wording')} ]})['passed']
    actual = bench.snapshot()
    assert actual.pop('task') == 'New wording'
    source.pop('task')
    assert actual == source


def test_paged_reads_are_exact_bounded_and_unicode_safe():
    from env_factory_v2.expressions import canonical
    value = {'a/b~c': ['多字节\\\"' * 10000]}
    bench = AuthorWorkbench(source={'environment': value})
    text, offset = '', 0
    while offset is not None:
        response = bench.call('read_contract', {'path': '/environment/a~1b~0c/0', 'offset': offset})
        assert response['passed'] and len(response['json_fragment']) <= 8192
        assert not response['complete']
        text += response['json_fragment']
        offset = response['next_offset']
    assert text == canonical(value['a/b~c'][0])
    assert json.loads(text) == value['a/b~c'][0]
    assert bench.snapshot() == {'environment': value}


@pytest.mark.parametrize('arguments', [
    {'path': '../secret', 'offset': 0}, {'path': '/environment/~2', 'offset': 0},
    {'path': '/environment/0', 'offset': 0}, {'path': '', 'offset': -1},
    {'path': '', 'offset': True}, {'path': '', 'offset': 10**9},
    {'path': '/environment/items/01', 'offset': 0},
    {'path': '/environment/items/-1', 'offset': 0},
    {'path': '', 'offset': 0, 'file': '/etc/passwd'}])
def test_reads_reject_invalid_paths_offsets_and_extra_access(arguments):
    bench = AuthorWorkbench(source={'environment': {'items': [1]}})
    original = bench.snapshot()
    assert not bench.call('read_contract', arguments)['passed']
    assert bench.snapshot() == original


def test_reads_consume_operation_budget_without_invalidating_checks():
    bench = AuthorWorkbench(source=direct(), max_operations=2)
    assert bench.call('check_contract', {})['passed']
    previous = bench.last_check
    assert bench.call('read_contract', {'path': '', 'offset': 0})['passed']
    assert bench.last_check is previous
    with pytest.raises(ContractError, match='budget exceeded'):
        bench.call('read_contract', {'path': '', 'offset': 0})


def test_fresh_draft_cannot_switch_to_flat_or_mixed_contract():
    bench=AuthorWorkbench()
    assert not bench.call('put_section',{'section':'rules','value_json':'[]'})['passed']
    assert not bench.call('put_section',{'section':'task','value_json':'"flat task"'})['passed']
    assert not bench.call('edit_contract',{'edits':[{'op':'add','path':'/tables','value_json':'[]'}]})['passed']
    assert bench.snapshot()=={}


def test_task_first_partial_definition_retains_format_after_resume():
    from test_business_definition import split
    environment,task=split(simple())
    bench=AuthorWorkbench(source={'task':task})
    assert not bench.call('put_section',{'section':'tables','value_json':'[]'})['passed']
    assert bench.call('put_section',{'section':'environment','value_json':json.dumps(environment)})['passed']
    assert bench.call('check_contract',{})['passed']


def test_explicit_historical_flat_import_remains_editable_without_mixing_formats():
    bench=AuthorWorkbench(source=direct())
    assert bench.call('check_contract',{})['passed']
    assert not bench.call('put_section',{'section':'environment','value_json':'{}'})['passed']
    assert bench.snapshot()==direct()
