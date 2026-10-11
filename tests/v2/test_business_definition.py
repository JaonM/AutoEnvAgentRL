import copy

import pytest

from env_factory_v2.business_definition import assemble_contract, compile_definition
from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import simple
from env_factory_v2.expressions import ContractError
from env_factory_v2.verification import expected_result, run_scenario


def split(source):
    environment_keys = {'tables', 'rules', 'tools', 'time', 'invariants', 'events'}
    environment = {k: copy.deepcopy(v) for k, v in source.items() if k in environment_keys}
    environment.update(version='1.0', id='shared_environment')
    task = {k: copy.deepcopy(v) for k, v in source.items()
            if k not in environment_keys | {'version', 'task'}}
    task.update(version='1.0', request=source['task'])
    return environment, task


def report_definition():
    environment, task = split(simple())
    task['answers'] = []
    task['reports'] = [{'id': 'ids', 'meaning': 'Matching records', 'query': 'find_items',
                        'arguments': {'size': {'ref': 'user.size'}}, 'columns': ['id'], 'weight': 1}]
    task['scenarios'][0]['answer']['ids']['map']['value'] = {'object': {'id': {'ref': 'row.id'}}}
    task['scenarios'][2]['answer']['ids'] = {'literal': [{'id': 'I1'}]}
    task['probes'][2]['answer']['ids'] = {'literal': [{'id': 'I1'}]}
    return environment, task


def test_split_preserves_existing_execution_contract():
    source = simple()
    environment, task = split(source)
    before = copy.deepcopy((environment, task))
    compiled, provenance = compile_definition(environment, task)
    assert compiled['identity'] == compile_contract(source)['identity']
    assert (environment, task) == before
    assert provenance['scope'] == 'compiled_definition_not_independent_business_acceptance'


@pytest.mark.parametrize('key', ['tables', 'rules', 'tools', 'time', 'events', 'invariants'])
def test_task_cannot_override_environment(key):
    environment, task = split(simple())
    task[key] = []
    with pytest.raises(ContractError, match='fields mismatch'):
        compile_definition(environment, task)


def test_reports_derive_rules_and_types_but_keep_execution_obligations():
    environment, task = report_definition()
    compiled, provenance = compile_definition(environment, task)
    assert provenance['derived_reports'] == ['ids']
    for scenario in compiled['source']['scenarios']:
        assert expected_result(scenario, run_scenario(compiled, scenario))
    assert compiled['source']['answers'][0]['schema']['items']['properties']['id'] == {'type': 'string'}


def test_changed_shared_rule_changes_query_and_reward_without_task_edit():
    environment, task = report_definition()
    first, _ = compile_definition(environment, task)
    environment['rules'][0]['predicate'] = {'literal': True}
    changed, _ = compile_definition(environment, task)
    scenario = task['scenarios'][0]
    assert expected_result(scenario, run_scenario(changed, scenario))
    stale = copy.deepcopy(scenario)
    stale['answer'] = {'ids': {'literal': [{'id': 'I1'}]}}
    assert run_scenario(first, stale)['outcome']['success']
    assert not run_scenario(changed, stale)['outcome']['success']


def test_multiple_tasks_share_environment_identity():
    environment, task = report_definition()
    _, first = compile_definition(environment, task)
    task['id'] = 'other_task'
    task['user']['facts']['size'] = 'L'
    compiled, second = compile_definition(environment, task)
    assert first['environment_identity'] == second['environment_identity']
    assert first['task_identity'] != second['task_identity']
    assert first['contract_identity'] != second['contract_identity']
    assert expected_result(task['scenarios'][0], run_scenario(compiled, task['scenarios'][0]))


def test_report_cannot_read_unpublished_columns():
    environment, task = report_definition()
    environment['tools'][0]['columns'] = ['id']
    task['reports'][0]['columns'] = ['size']
    with pytest.raises(ContractError, match='published query columns'):
        assemble_contract(environment, task)


def test_unfiltered_report_uses_full_collection():
    environment, task = report_definition()
    environment['tools'][0]['rule_ids'] = []
    compiled, _ = compile_definition(environment, task)
    scenario = task['scenarios'][0]
    assert expected_result(scenario, run_scenario(compiled, scenario))
    stale = copy.deepcopy(scenario)
    stale['answer'] = {'ids': {'literal': [{'id': 'I1'}]}}
    assert not run_scenario(compiled, stale)['outcome']['success']


def test_cli_emits_reproducible_sources_without_qualification(tmp_path, monkeypatch):
    import json
    import sys
    from env_factory_v2.__main__ import main

    environment, task = report_definition()
    env_path, task_path = tmp_path / 'environment.json', tmp_path / 'task.json'
    env_path.write_text(json.dumps(environment))
    task_path.write_text(json.dumps(task))
    output = tmp_path / 'compiled'
    monkeypatch.setattr(sys, 'argv', ['v2', 'compile-definition', '--environment', str(env_path),
                                    '--task', str(task_path), '--output', str(output)])
    assert main() == 0
    provenance = json.loads((output / 'compilation.json').read_text())
    contract = compile_contract(json.loads((output / 'contract.json').read_text()))
    assert provenance['contract_identity'] == contract['identity']
    assert not (output / 'verification.json').exists()
    with pytest.raises(FileExistsError):
        main()


def test_workbench_checks_definitions_and_atomic_nested_repairs():
    import json
    from env_factory_v2.author_workbench import AuthorWorkbench

    environment, task = report_definition()
    bench = AuthorWorkbench()
    for key, value in [('environment', environment), ('task', task)]:
        assert bench.call('put_section', {'section': key, 'value_json': json.dumps(value)})['passed']
    result = bench.call('check_contract', {})
    assert result['passed']
    assert result['compilation']['derived_reports'] == ['ids']
    assert bench.call('edit_contract', {'edits': [{'op': 'add', 'path': '/task/rules',
                                                'value_json': '[]'}]})['passed']
    result = bench.call('check_contract', {})
    assert not result['passed']
    assert 'fields mismatch' in result['error']


def test_pipeline_retains_definition_on_bounded_repair(tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock
    from env_factory_v2.pipeline import build

    environment, task = report_definition()
    definition = {'environment': environment, 'task': task}
    repaired = copy.deepcopy(definition)
    repaired['task']['id'] = 'repaired_task'
    agent = Mock()
    agent.generate.return_value = (repaired, {'unit_test': True})
    challenge = Mock(return_value={'passed': False, 'instruction': 'unit-test rejection'})
    monkeypatch.setattr('env_factory_v2.pipeline.challenge', challenge)
    output = tmp_path / 'build'
    result = build({'category': task['category']}, agent=agent, source=definition,
                   output=output, repairs=1, docker=False, tool_interference=2)
    assert not result['qualified']
    assert challenge.call_count == 2
    assert agent.generate.call_args.kwargs['feedback']['previous_source'] == definition
    assert json.loads((output / 'attempt-1/definition.json').read_text()) == definition
    source = json.loads((output / 'attempt-1/source.json').read_text())
    assert source['version'] == '2.0'
    assert len(source['tools']) > len(environment['tools'])
    receipt = json.loads((output / 'attempt-1/compilation.json').read_text())
    assert receipt['contract_identity'] == compile_definition(environment, task)[0]['identity']
    assert receipt['contract_identity'] != compile_contract(source)['identity']


def test_generation_requests_separate_definition_and_preserves_nested_edits(tmp_path):
    import json
    from unittest.mock import Mock
    from env_factory_v2.code_agent import NativeCodeAgent

    environment, task = report_definition()
    definition = {'environment': environment, 'task': task}
    agent = NativeCodeAgent(base_url='http://127.0.0.1:9999/v1',
                            token='test-only-token-000000000000000000', evidence_root=tmp_path)
    agent.ask = Mock(return_value=({'source_json': json.dumps(definition)}, {'unit_test': True}))
    generated, _ = agent.generate({'goal': task['request']})
    assert generated == definition
    assert '"environment": ENVIRONMENT' in agent.ask.call_args.args[0]
    agent.ask.return_value = ({'edits': [{'op': 'replace', 'path': '/task/id',
                                        'value_json': '"repaired"'}]}, {'unit_test': True})
    repaired, _ = agent.generate({'goal': task['request']}, feedback={'previous_source': generated})
    assert repaired['task']['id'] == 'repaired'
    assert repaired['environment'] == environment
    assert generated['task']['id'] == task['id']
