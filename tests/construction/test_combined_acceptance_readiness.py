"""Shared execution must remove duplicate scenarios without weakening gates."""
import copy
import json
import sys
import types

import pytest

from env_factory import sandbox_runtime
from env_factory.generation.agent_authoring import compile_source
from scripts.sandbox import generate_sandbox_scaffold as scaffold
from scripts.sandbox import validate_training_readiness as readiness
from tests.generation.test_agent_authoring import fixture


@pytest.fixture
def generated(tmp_path, monkeypatch):
    source, request = fixture('direct_response')
    for scenario in source['scenarios']:
        scenario['steps'].insert(0, {'operation': 'reset',
            'body': {'episode_id': scenario['scenario_id'], 'seed': 17}, 'expected_status': 200})
        scenario['steps'].append({'operation': 'reward', 'expected_status': 200})
    artifact = compile_source(source, root=tmp_path, request=request, structural_preview=True)
    contract = {**artifact, 'artifacts': artifact}
    for name in ('task.json', 'BUILD_CONTRACT.json'):
        (tmp_path / name).write_text(json.dumps(contract))
    scaffold.generate(tmp_path)
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK', '1')
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'test-key')
    monkeypatch.setitem(sys.modules, 'sandbox_runtime', sandbox_runtime)
    hooks = types.ModuleType('task_impl')
    exec(scaffold.TASK_IMPL_SOURCE, hooks.__dict__)
    monkeypatch.setitem(sys.modules, 'task_impl', hooks)
    app = types.ModuleType('app')
    app.__file__ = str(tmp_path / 'app.py')
    exec(scaffold.APP_SOURCE, app.__dict__)
    monkeypatch.setitem(sys.modules, 'app', app)
    acceptance = types.ModuleType('acceptance')
    acceptance.__file__ = str(tmp_path / 'acceptance_runner.py')
    exec(scaffold.ACCEPTANCE_RUNNER_SOURCE, acceptance.__dict__)
    return tmp_path, acceptance


def test_combined_execution_preserves_gates_and_removes_one_full_scenario_pass(generated, monkeypatch):
    root, acceptance = generated
    executions = []
    original = sandbox_runtime.AcceptanceScenarioRunner.run
    def run(self, scenario):
        executions.append(scenario.get('scenario_id'))
        return original(self, scenario)
    monkeypatch.setattr(sandbox_runtime.AcceptanceScenarioRunner, 'run', run)
    assert acceptance.main() == 0
    separate = readiness.validate(root)
    separate_count = len(executions)
    executions.clear()
    combined = readiness.validate_with_acceptance(root)
    assert separate['training_ready'] is True, separate
    assert combined['training_ready'] is True, combined
    scenarios = json.loads((root / 'task.json').read_text())['acceptance_contract']['executable_scenarios']
    shared_kinds = {'goal_success', 'goal_failure', 'noise_selection', 'counterfactual'}
    assert len(executions) == separate_count - sum(s.get('kind') in shared_kinds for s in scenarios)
    assert executions.count('success') == 3  # Baseline plus two independent determinism runs.
    for gate in ('rewards', 'determinism', 'observation_scan', 'runtime_state'):
        assert combined['evidence'][gate] == separate['evidence'][gate]
    assert combined['evidence']['scenario_execution']['shared_scenarios'] == len(scenarios)


def test_shared_runs_reject_changed_contract_and_keep_negative_reward_checks(generated):
    root, acceptance = generated
    runs = []
    assert acceptance.main(scenario_results=runs) == 0
    changed = copy.deepcopy(runs)
    changed[0][0]['scenario_id'] = 'different-contract'
    with pytest.raises(ValueError, match='current contract'):
        readiness.validate(root, acceptance_runs=changed)
    # Even shared outcomes are subjected to readiness's own reward separation check.
    for scenario, result in runs:
        if scenario.get('kind') == 'goal_failure':
            for step in result['history']:
                if step['operation'] == 'reward':
                    step['body']['reward'] = 1
    assert 'reward_separation' in readiness.validate(root, acceptance_runs=runs)['failed_gates']


def test_combined_execution_does_not_accept_stale_pass_files(generated):
    root, _ = generated
    (root / 'acceptance_result.json').write_text('{"business_acceptance":"passed"}')
    contract = json.loads((root / 'BUILD_CONTRACT.json').read_text())
    contract['acceptance_contract']['executable_scenarios'][0]['steps'][0]['expected_status'] = 418
    (root / 'BUILD_CONTRACT.json').write_text(json.dumps(contract))
    result = readiness.validate_with_acceptance(root)
    assert result['training_ready'] is False
    assert result['failed_gates'] == ['business_acceptance']
    assert json.loads((root / 'acceptance_result.json').read_text())['business_acceptance'] == 'failed'


def test_matching_independent_failures_cannot_overrule_a_shared_pass(generated, monkeypatch):
    root, _ = generated
    original = sandbox_runtime.AcceptanceScenarioRunner.run
    success_runs = []
    def run(self, scenario):
        result = original(self, scenario)
        if scenario.get('kind') == 'goal_success':
            success_runs.append(True)
            if len(success_runs) > 1:
                for step in result['history']:
                    if step['operation'] == 'reward':
                        step['body']['reward'] = 0
        return result
    monkeypatch.setattr(sandbox_runtime.AcceptanceScenarioRunner, 'run', run)
    result = readiness.validate_with_acceptance(root)
    assert result['evidence']['rewards']['success'] == 1
    assert result['evidence']['independent_rewards'] == [0, 0]
    assert 'determinism' in result['failed_gates']
