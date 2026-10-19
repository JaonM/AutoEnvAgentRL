"""Bounded repair executes real gates and frozen external behavioral checks; no model calls."""
import json
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import FakeRunner, LIBRARY, library_request
from test_world import world_request

from env_factory.authoring import pipeline
from env_factory.authoring.codex import InfrastructureError
from env_factory.core import answers, spec
from env_factory.production import world
from env_factory.production.acceptance import Suite, repair_bundle


CHECKER = '''import copy, json, sys
from pathlib import Path
from env_factory.core.sandbox import load_module
from env_factory.runtime import BusinessError
folder, out = map(Path, sys.argv[1:3])
env = load_module(folder / 'env.py', role='env')
db = json.loads((folder / 'request.json').read_text())['tables']
before = copy.deepcopy(db)
oracle = json.loads(Path(sys.argv[3]).read_text())
results = []
for book_id in oracle['ids']:
    try:
        env.make_tools()['get_waitlist'](db, 10, book_id)
        message = None
    except BusinessError as error:
        message = str(error)
    results.append(message == oracle['message'].format(book_id=book_id) and db == before)
(out / 'report.json').write_text(json.dumps({'passed': all(results), 'results': results,
                                          'expected_error': oracle['message']}))
sys.exit(0 if all(results) else 1)
'''


@pytest.fixture
def case(tmp_path):
    source = tmp_path / 'source'
    spec.materialize(library_request(), source, answers.load_schema(LIBRARY / 'answer_schema.json'))
    good = (LIBRARY / 'env.py').read_text()
    bad = good.replace("f'{label} {record_id} not found'", "'internal_not_found'")
    assert good != bad
    (source / 'env.py').write_text(bad)
    shutil.copyfile(LIBRARY / 'verifier.py', source / 'verifier.py')
    checker, oracle = tmp_path / 'checker.py', tmp_path / 'oracle.json'
    checker.write_text(CHECKER)
    oracle.write_text(json.dumps({'ids': ['missing-A', 'missing-B'], 'message': 'book {book_id} not found'}))
    config = {'checks': [{'name': 'public_errors', 'command': [sys.executable, str(checker), '{bundle}',
                                                           '{evidence}', str(oracle)],
                          'protected_files': [str(checker), str(oracle)]}]}
    return source, good, config, oracle


def test_repairs_real_behavior_without_changing_truth_and_retains_failure(case, tmp_path):
    source, good, config, _ = case
    def runner(folder, prompt, **options):
        assert options['role'] == 'acceptance_repair'
        rules = json.loads((folder / 'context/platform_rules.json').read_text())
        assert 'ast' not in rules['allowed_imports'] and 'operator' in rules['allowed_imports']
        assert rules['global_and_nonlocal_statements_allowed'] is False
        assert '-m env_factory.checks env' in rules['self_check']
        assert not json.loads((folder / 'acceptance_feedback.json').read_text())['external']['passed']
        (folder / 'env.py').write_text(good)
    result = repair_bundle(source, tmp_path / 'repaired', runner=runner, suite=config)
    assert result['accepted'] and result['repairs_used'] == 1
    assert result['rounds'][0]['gates_passed'] and not result['rounds'][0]['external']['passed']
    assert result['rounds'][1]['gates_passed'] and result['rounds'][1]['external']['passed']
    for name in ('verifier.py', 'request.json', 'answer_schema.json'):
        assert (Path(result['bundle']) / name).read_bytes() == (source / name).read_bytes()
    assert (source / 'env.py').read_text() != good


@pytest.mark.parametrize('violation', ['import ast\n',
    'def closure():\n    position = 0\n    def advance():\n        nonlocal position\n        position += 1\n'])
def test_delivered_self_check_runs_in_isolated_candidate_and_catches_runtime_restrictions(case, tmp_path, violation):
    source, good, config, _ = case
    def runner(folder, prompt, **options):
        rules = json.loads((folder / 'context/platform_rules.json').read_text())
        command = shlex.split(rules['self_check'])
        assert Path(command[0]).is_absolute()
        (folder / 'env.py').write_text(violation + good)
        failed = subprocess.run(command, cwd=folder, capture_output=True, text=True)
        report = json.loads((folder / 'check-env.json').read_text())
        assert failed.returncode == 1 and not report['passed']
        assert any('not allowed' in p for p in report['problems'])
        (folder / 'env.py').write_text(good)
        passed = subprocess.run(command, cwd=folder, capture_output=True, text=True)
        assert passed.returncode == 0, passed.stdout + passed.stderr
        return {'status': 'exited', 'usage': {}, 'self_check_executed': True}
    result = repair_bundle(source, tmp_path / 'repaired', runner=runner, suite=config)
    assert result['accepted'] and result['repairs_used'] == 1
    evidence = json.loads((tmp_path / 'repaired/round-01/repair_session.json').read_text())
    assert evidence['self_check_executed']


@pytest.mark.parametrize('target', ['verifier', 'oracle'])
def test_repair_cannot_modify_independent_truth(case, tmp_path, target):
    source, good, config, oracle = case
    def runner(folder, prompt, **options):
        (folder / 'env.py').write_text(good)
        path = folder / 'verifier.py' if target == 'verifier' else oracle
        path.write_text(path.read_text() + '\n ')
    result = repair_bundle(source, tmp_path / 'repaired', runner=runner, suite=config)
    assert not result['accepted'] and result['failure_class'] == 'acceptance_tampered'


@pytest.mark.parametrize('failure_class', ['specification', 'teacher', 'infrastructure'])
def test_nonimplementation_failure_never_triggers_environment_repair(case, tmp_path, failure_class):
    source, _, config, _ = case
    config['checks'][0]['failure_class'] = failure_class
    def runner(*args, **kwargs):
        pytest.fail('wrong repair class')
    result = repair_bundle(source, tmp_path / 'repaired', runner=runner, suite=config)
    assert result['failure_class'] == failure_class and result['repairs_used'] == 0


def test_repair_budget_and_no_progress_are_terminal(case, tmp_path):
    source, _, config, _ = case
    calls = []
    def runner(*args, **kwargs):
        calls.append(kwargs['role'])
    exhausted = repair_bundle(source, tmp_path / 'zero', runner=runner, suite=config, repairs=0)
    assert exhausted['failure_class'] == 'repair_budget_exhausted' and not calls
    stalled = repair_bundle(source, tmp_path / 'stalled', runner=runner, suite=config, repairs=2)
    assert stalled['failure_class'] == 'no_progress' and len(calls) == 1


def test_unserved_model_call_refunds_shared_budget(case, tmp_path):
    source, _, config, _ = case
    budget = [1]
    def spend():
        budget[0] -= 1
        return True
    def refund():
        budget[0] += 1
    def runner(*args, **kwargs):
        raise InfrastructureError('provider unavailable')
    result = repair_bundle(source, tmp_path / 'repaired', runner=runner, suite=config, spend=spend, refund=refund)
    assert result['failure_class'] == 'infrastructure'
    assert result['repairs_used'] == 0 and budget == [1]


def test_invalid_checker_json_is_infrastructure_not_a_business_defect(case, tmp_path):
    source, _, config, _ = case
    checker = Path(config['checks'][0]['protected_files'][0])
    checker.write_text("import sys\nfrom pathlib import Path\n(Path(sys.argv[2])/'report.json').write_text('[]')\n")
    verdict = Suite(config).evaluate(source, tmp_path / 'evidence')
    assert not verdict['passed'] and verdict['failure_class'] == 'infrastructure'


def test_build_entry_runs_external_acceptance_and_uses_shared_budget(case, tmp_path):
    source, good, config, _ = case
    base = FakeRunner(env_source=(source / 'env.py').read_text())
    def runner(folder, prompt, **options):
        if options['role'] == 'acceptance_repair':
            (folder / 'env.py').write_text(good)
            return {'status': 'exited'}
        return base(folder, prompt, **options)
    result = pipeline.build(library_request(), tmp_path / 'build', runner=runner, acceptance_suite=config)
    assert result['training_ready'], result
    assert result['stages']['external_acceptance']['accepted'] and result['repairs_used'] == 1


def test_world_repairs_invalid_specification_and_passes_remaining_budget(tmp_path):
    calls = []
    def runner(folder, prompt, **options):
        calls.append(options['role'])
        request = world_request()
        if options['role'] == 'world_author':
            request['tables']['books'] = request['tables']['books'][:3]
        (folder / 'request.json').write_text(json.dumps(request))
    with patch.object(world.generate, 'write_opening', return_value=('Renew my loans.', {}, [])), \
         patch.object(world.agent_policy, 'write', return_value=('Library policy', [])), \
         patch.object(pipeline, 'build', return_value={'training_ready': True, 'repairs_used': 0}) as build:
        result = world.build_world({'name': 'library'}, tmp_path / 'world', runner=runner, client=None,
                                   python=sys.executable, language='en', index=0, repairs=2)
    assert calls == ['world_author', 'world_repair']
    assert result['spec_repairs_used'] == result['total_repairs_used'] == 1
    assert build.call_args.kwargs['repairs'] == 1
    first = json.loads((tmp_path / 'world/specification_attempts/00/verdict.json').read_text())
    assert not first['passed']


def test_world_specification_repair_is_bounded(tmp_path):
    calls = []
    def runner(folder, prompt, **options):
        calls.append(options['role'])
        (folder / 'request.json').write_text('{}')
    result = world.build_world({'name': 'library'}, tmp_path / 'world', runner=runner, client=None,
                               python=sys.executable, language='en', index=0, repairs=1)
    assert not result['training_ready'] and result['spec_repairs_used'] == 1
    assert calls == ['world_author', 'world_repair']


@pytest.mark.parametrize('tamper', [True, False])
def test_world_preserves_contract_and_does_not_repair_a_malformed_auditor_reply(tmp_path, tamper):
    from test_public_contracts import CONTRACT
    calls = []
    def runner(folder, prompt, **options):
        calls.append(options['role'])
        (folder / 'request.json').write_text(json.dumps(world_request()))
        if tamper:
            (folder / 'public_contract.json').write_text('{}')
    with patch('env_factory.production.contracts.review', return_value={
            'passed': False, 'stage': 'audit_invalid', 'problems': ['bad verdict JSON']}):
        result = world.build_world({'name': 'library', 'public_contract': CONTRACT}, tmp_path / 'world',
                                   runner=runner, client=None, python=sys.executable, language='en', index=0)
    assert result['failure_class'] == ('public_contract_modified' if tamper else 'audit_invalid')
    assert calls == ['world_author']


def test_checker_cannot_change_the_candidate_after_qualification(case, tmp_path):
    source, _, config, _ = case
    checker = Path(config['checks'][0]['protected_files'][0])
    checker.write_text("import sys\nfrom pathlib import Path\n"
                       "(Path(sys.argv[1])/'verifier.py').write_text('# invalid replacement')\n"
                       "(Path(sys.argv[2])/'report.json').write_text('{\"passed\":true}')\n")
    result = repair_bundle(source, tmp_path / 'repaired', runner=None, suite=config)
    assert not result['accepted'] and result['failure_class'] == 'acceptance_tampered'


def test_caller_behavior_contract_drives_bounded_repair_and_stays_frozen(case, tmp_path):
    from env_factory.certify import behavior
    source, good, config, _ = case
    request=json.loads((source/'request.json').read_text())
    contract={'version':1,'cases':[{'id':'unseen-book','data':request['tables'],
        'steps':[{'tool':'get_waitlist','args':{'book_id':'unseen-contract-id'},
                  'error':'book unseen-contract-id not found'}]}]}
    request.update(behavior_contract=contract,behavior_contract_sha256=behavior.digest(contract))
    (source/'request.json').write_text(json.dumps(request))
    def runner(folder, prompt, **options):
        feedback=json.loads((folder/'acceptance_feedback.json').read_text())
        assert not feedback['gates']['behavior']['passed']
        (folder/'env.py').write_text(good)
    result=repair_bundle(source,tmp_path/'fixed',runner=runner,suite=config)
    assert result['accepted'] and result['repairs_used']==1
    assert result['rounds'][1]['gates']['behavior']['passed']
    assert json.loads((Path(result['bundle'])/'request.json').read_text())['behavior_contract']==contract


@pytest.mark.parametrize('tamper',[False,True])
def test_standard_pipeline_receives_behavior_failures_and_protects_truth(case,tmp_path,tamper):
    from env_factory.certify import behavior
    source,good,_,_=case
    request=library_request()
    contract={'version':1,'cases':[{'id':'independent-missing-book','data':request['tables'],
        'steps':[{'tool':'get_waitlist','args':{'book_id':'unseen-contract-id'},'error':'book unseen-contract-id not found'}]}]}
    request.update(behavior_contract=contract,behavior_contract_sha256=behavior.digest(contract))
    def repair(folder):
        feedback=json.loads((folder/'gate_report.json').read_text())
        assert not feedback['behavior']['passed']
        (folder/'env.py').write_text(good)
        if tamper:
            verifier=folder/'verifier.py';verifier.write_text(verifier.read_text()+'\n')
    runner=FakeRunner(env_source=(source/'env.py').read_text(),repair=repair)
    result=pipeline.build(request,tmp_path/'build',runner=runner)
    if tamper:assert result['failure_class']=='acceptance_tampered' and not result['training_ready']
    else:assert result['training_ready'] and result['repairs_used']==1
