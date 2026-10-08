import copy
import json

import pytest

from rl.rollout.budgets import TaskBudgetError, admit, execution_budget
from rl.rollout.worker import rollout
from tests.rl.test_native_tools import NativeRuntime, NativePolicy, NativeDecoder


SECRET = 'REFERENCE-ONLY-CONTROL-DO-NOT-SAMPLE'


def budget():
    return {'contract': {'version': '1.0', 'profile': 'long_horizon_planning',
        'minimum_model_actions': 1, 'reference_count': 1}, 'dialogue_tool': None, 'retry_tool': 'retry_tool_call',
        'references': [{'scenario_id': 'ref', 'records': [
            {'kind': 'tool', 'name': 'lookup', 'arguments': {'key': '甲'}, 'request_group': 0,
             'status': 200, 'result': {'records': [SECRET]}},
            {'kind': 'tool', 'name': 'lookup', 'arguments': {'key': '甲'}, 'request_group': 0,
             'status': 200, 'result': {'records': [SECRET]}}, {'kind': 'answer', 'content': SECRET}]}]}


class CountingPolicy(NativePolicy):
    def encode(self, messages, *, tools):
        return list(range(len(json.dumps([messages, tools], ensure_ascii=False))))

    def sample(self, messages, *args, **kwargs):
        assert SECRET not in json.dumps(messages), 'trainer reference leaked to sampling'
        return super().sample(messages, *args, **kwargs)


class CountingBatchedPolicy(CountingPolicy):
    def batch_decoder(self):
        return NativeDecoder(self)


class BudgetRuntime(NativeRuntime):
    def __init__(self, root):
        super().__init__(root)
        value = budget()
        self.task['task_spec'] = {'difficulty_requirements': {'profile': 'long_horizon_planning'},
                                  'execution_budget': value['contract']}
        self.task['artifacts'] = {'generation_pipeline': {'verification': {'policy_budget_references': value['references']}}}

    def snapshot(self):
        return {**copy.deepcopy(self.__dict__), 'names': sorted(self.names)}

    def restore(self, state):
        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(state))
        self.names = set(state['names'])


def config():
    return {'max_steps': 2, 'max_tokens': 16, 'max_context': 20000, 'rollout_group': 1,
            'algorithm': 'grpo', 'rollout_timeout': 10}


def test_reference_capacity_is_reported_without_rejecting_shorter_native_policy():
    env = BudgetRuntime(None)
    policy = CountingPolicy()
    events = []
    result = rollout(policy, env, config(), 17, observe=lambda event, **data: events.append((event, data)))
    report = result['budget_admission']
    assert result['final_reward'] == 1
    assert report['reference_serial_actions'] == 3 > config()['max_steps']
    assert report['reference_capacity_fits'] is False
    assert report['recommended_max_context'] > report['current_prompt_tokens']
    assert SECRET not in json.dumps([env.messages, report, events], ensure_ascii=False)
    assert any(event == 'budget_admission' for event, _ in events)


def test_actual_initial_context_and_missing_evidence_fail_before_sampling():
    env = BudgetRuntime(None)
    with pytest.raises(TaskBudgetError, match='CONTEXT_BUDGET_INCOMPATIBLE'):
        rollout(CountingPolicy(), env, {**config(), 'max_context': 10}, 17)
    assert not any(row['path'].startswith('/v1/tools/') for row in env.trace)
    env.task['task_spec'].pop('execution_budget')
    with pytest.raises(TaskBudgetError, match='EVIDENCE_MISSING'):
        rollout(CountingPolicy(), env, config(), 17)


def test_tokenizer_sizing_does_not_mutate_live_messages_or_reference_records():
    env = BudgetRuntime(None); env.reset(17)
    original = copy.deepcopy(env.messages)
    value = execution_budget(env); before = copy.deepcopy(value)
    admit(CountingPolicy(), env.messages, {'tools': env.tools}, config(), value)
    assert env.messages == original and value == before


@pytest.mark.parametrize('continuation', [False, True])
@pytest.mark.parametrize('policy_type', [CountingPolicy, CountingBatchedPolicy])
def test_parallel_rpc_and_durable_replies_preserve_budget_admission(tmp_path, continuation, policy_type):
    from rl.rollout.scheduler import EnvironmentPool, parallel_rollouts
    pool = EnvironmentPool(str(tmp_path), 1, factory='tests.rl.test_task_budgets:BudgetRuntime')
    try:
        result = parallel_rollouts(policy_type(), pool, config(), 17,
            **({'continuation': tmp_path / 'continuation'} if continuation else {}))[0]
        assert result['final_reward'] == 1
        assert result['budget_admission']['reference_serial_actions'] == 3
        assert SECRET not in json.dumps(result, ensure_ascii=False)
    finally:
        pool.close()


def test_resumed_durable_step_rechecks_changed_context_budget(tmp_path):
    import threading
    from rl.rollout.scheduler import EnvironmentPool, parallel_rollouts
    stop = threading.Event()
    def observe(event, **fields):
        if event == 'action':
            stop.set()
    pool = EnvironmentPool(str(tmp_path), 1, factory='tests.rl.test_task_budgets:BudgetRuntime')
    try:
        with pytest.raises(InterruptedError):
            parallel_rollouts(CountingPolicy(), pool, config(), 17, stop=stop, observe=observe,
                              continuation=tmp_path / 'journal')
    finally:
        pool.close()
    pool = EnvironmentPool(str(tmp_path), 1, factory='tests.rl.test_task_budgets:BudgetRuntime')
    try:
        with pytest.raises(TaskBudgetError, match='CONTEXT_BUDGET_INCOMPATIBLE'):
            parallel_rollouts(CountingPolicy(), pool, {**config(), 'max_context': 10}, 17,
                              continuation=tmp_path / 'journal')
    finally:
        pool.close()
