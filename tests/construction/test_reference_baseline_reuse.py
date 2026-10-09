"""Candidate probes share current stable baselines, never changed counterfactuals."""
import copy

import pytest

from scripts.sandbox import validate_agentic_training_value as gate


class ProbeApp:
    def __init__(self, *, frozen=False):
        self.initial = {'facts': [{'id': 1, 'left': 'A', 'right': 'B'}]}
        self.frozen = frozen
        self.baseline_rewards = self.changed_rewards = 0
        self.tables = copy.deepcopy(self.initial)

    def business_snapshot(self):
        return copy.deepcopy(self.tables)

    def mutate_business_state(self, mutation):
        self.tables[mutation['table']][0].update(mutation['changes'])

    def handle(self, method, path, body=None, headers=None):
        if path == '/v1/reset':
            self.tables = copy.deepcopy(self.initial)
            return 200, {}, {}
        if path == '/v1/tools/read':
            return 200, {'records': self.business_snapshot()['facts']}, {}
        if path == '/v1/agent_response':
            return 200, {}, {}
        if path == '/v1/reward':
            initial, actual = self.initial['facts'][0], self.tables['facts'][0]
            changed = actual != initial
            if changed:
                self.changed_rewards += 1
            else:
                self.baseline_rewards += 1
            reward = (1.0 if self.frozen else
                      .4 + .3 * (actual['left'] == initial['left']) + .3 * (actual['right'] == initial['right']))
            return 200, {'reward': reward}, {}
        raise AssertionError(path)


SCENARIO = {'steps': [{'operation': 'reset'},
    {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {}},
    {'operation': 'agent_response', 'content': 'A B'}, {'operation': 'reward'}]}
MUTATIONS = [{'table': 'facts', 'selector': {'id': 1}, 'changes': {field: 'changed'}}
             for field in ('left', 'right')]


@pytest.mark.parametrize('frozen', [False, True])
def test_one_baseline_pair_serves_candidates_without_skipping_changed_repeats(frozen):
    app = ProbeApp(frozen=frozen)
    result = gate.probe_reference_data_reward_candidates(app, {}, SCENARIO, MUTATIONS, .8)
    assert result['proved'] is (not frozen)
    assert app.baseline_rewards == 2
    assert app.changed_rewards == (4 if frozen else 6)
    attempts = result['candidate_attempts']
    assert len(attempts) == (2 if frozen else 3)
    dependencies = [item['result']['baseline_dependency'] for item in attempts]
    assert [d['reused'] for d in dependencies] == ([False, True] if frozen else [False, True, True])
    assert len({d['identity'] for d in dependencies}) == 1
    assert all(len(item['result']['changed_rewards']) == 2 for item in attempts)
    if not frozen:
        assert result['combined_counterfactual']
        assert result['changed_rewards'] == pytest.approx([.4, .4])


@pytest.mark.parametrize('change', ['initial_data', 'scenario', 'evaluator', 'application'])
def test_changed_dependency_requires_fresh_baseline_pair(monkeypatch, change):
    app, baselines = ProbeApp(), {}
    first = gate.probe_reference_data_reward_sensitivity(app, {}, SCENARIO, MUTATIONS[0], .8,
                                                        _baselines=baselines)
    scenario = copy.deepcopy(SCENARIO)
    if change == 'initial_data':
        app.initial['facts'][0]['left'] = 'new initial'
    elif change == 'scenario':
        scenario['steps'][-2]['content'] = 'changed explanation'
    elif change == 'evaluator':
        monkeypatch.setenv('SANDBOX_LLM_MODEL', 'changed-provider-model')
    else:
        app = ProbeApp()
    second = gate.probe_reference_data_reward_sensitivity(app, {}, scenario, MUTATIONS[1], .8,
                                                         _baselines=baselines)
    assert app.baseline_rewards == (2 if change == 'application' else 4)
    assert second['baseline_dependency']['reused'] is False
    assert first['baseline_dependency']['identity'] != second['baseline_dependency']['identity']


def test_unstable_baseline_is_not_shared(monkeypatch):
    app, baselines = ProbeApp(), {}
    original = app.handle
    def handle(*args, **kwargs):
        response = original(*args, **kwargs)
        if args[1] == '/v1/reward' and app.baseline_rewards == 2:
            response[1]['reward'] = 0
        return response
    monkeypatch.setattr(app, 'handle', handle)
    result = gate.probe_reference_data_reward_sensitivity(app, {}, SCENARIO, MUTATIONS[0], .8,
                                                         _baselines=baselines)
    assert result['proved'] is False
    assert result['baseline_rewards'] == [1, 0]
    assert not baselines
    assert app.changed_rewards == 0
