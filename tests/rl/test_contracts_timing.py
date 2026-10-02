import copy
import pytest
from rl.contracts import attach_contract, validate_contract, TrainingSegment, create_runtime
from rl.timing import StageTimes


def group():
    return {'task_id': 'sandbox', 'dataset_index': 3, 'policy_version': 2, 'episodes': [
        {'actions': [{'prompt': [1, 2], 'tokens': [3, 4], 'old_logp': [-.2, -.7]}]}]}


def test_segment_identity_and_behavior_tokens_are_preserved():
    value = group()
    before = copy.deepcopy(value['episodes'][0]['actions'][0])
    attach_contract(value)
    validate_contract(value)
    action = value['episodes'][0]['actions'][0]
    segment = TrainingSegment.from_action(action)
    assert segment.prompt == before['prompt'] and segment.tokens == before['tokens']
    assert segment.behavior_logprobs == before['old_logp']
    assert segment.loss_mask == [1, 1]
    action['policy_version'] = 99
    with pytest.raises(ValueError, match='identity'):
        validate_contract(value)


def test_stage_timing_classifies_external_wait_without_counting_it_as_gpu_work():
    times = StageTimes()
    with times.measure('actor_optimizer_step'):
        pass
    times.ingest_rollout({'episodes': [{'actions': [{'generation_compute_seconds': .1, 'verification_seconds': .02}],
        'trace': [{'path': '/v1/user_simulator', 'seconds': 3.}, {'path': '/v1/tools/search', 'seconds': .4},
                  {'path': '/v1/reward', 'seconds': .03}]}]})
    restored = StageTimes(times.state())
    assert restored.seconds['rollout_user_simulator'] == 3.
    assert restored.seconds['rollout_tool'] == .4
    assert restored.seconds['rollout_generation_compute'] == .1
    assert restored.counts['actor_optimizer_step'] == 1


class ToyRuntime:
    def __init__(self, sandbox):
        self.messages = []
        self.terminated = False
    def reset(self, seed):
        self.messages = [{'role': 'user', 'content': str(seed)}]
        return self.messages
    def step(self, text):
        self.terminated = True
        return 1., True
    def finish(self):
        return {'final_reward': 1., 'terminated': True}
    def close(self): pass


def test_runtime_factory_is_used_inside_isolated_environment_process():
    from rl.parallel_rollout import EnvironmentPool
    pool = EnvironmentPool('test-only', 1, factory=f'{__name__}:ToyRuntime')
    try:
        connection, _ = pool.workers[0]
        connection.send(('reset', 73))
        assert connection.poll(5)
        assert connection.recv()['messages'][0]['content'] == '73'
        connection.send(('step', 'anything'))
        assert connection.poll(5)
        assert connection.recv()['terminated'] is True
    finally:
        pool.close()
