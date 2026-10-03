import json
from types import SimpleNamespace

import pytest

from rl.chat import completion
from rl.environment import SandboxEpisode
from rl.parallel_rollout import EnvironmentPool, parallel_rollouts
from rl.rollout_worker import rollout


TOOLS = [{'type': 'function', 'function': {'name': 'lookup', 'description': '查询数量',
    'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}}}]
CALL = '<tool_call>\n{"name":"lookup","arguments":{"key":"甲"}}\n</tool_call>'


def tokenizer(text):
    def decode(tokens, *, skip_special_tokens):
        assert not skip_special_tokens, 'native delimiters must survive decoding'
        assert tokens == [1], 'strip EOS without dropping any generated content'
        return text
    return SimpleNamespace(eos_token_ids={2}, decode=decode, think_start='<think>', think_end='</think>',
        tool_call_start='<tool_call>', tool_call_end='</tool_call>', tool_parser=lambda body, tools: json.loads(body))


class NativeRuntime(SandboxEpisode):
    def __init__(self, root):
        self.task = {'task': '查询甲的数量'}

    def request(self, method, path, body=None):
        self.trace.append({'path': path, 'request': body})
        if path == '/v1/reset':
            return 200, {'reward_mode': 'episode_end'}
        if path == '/v1/tools':
            return 200, {'tools': TOOLS}
        if path == '/v1/observation':
            return 200, {'ready': True}
        if path == '/v1/tools/lookup':
            return 200, {'quantity': 7}
        if path == '/v1/agent_response':
            return 200, {}
        if path == '/v1/user_simulator':
            assert body['messages'] == [
                {'role': 'user', 'content': '查询甲的数量'}, {'role': 'assistant', 'content': '7件'}]
            return 200, {'user_query': '收到', 'should_end': True}
        raise AssertionError(path)

    def finish(self):
        return {'final_reward': 1., 'terminated': self.terminated, 'messages': self.messages, 'trace': self.trace}

    def close(self):
        pass


def test_native_call_result_and_plain_user_response():
    env = NativeRuntime(None)
    env.reset(42)
    assert env.tools == TOOLS
    assert 'lookup' not in env.messages[0]['content']  # schemas travel separately
    original = [1, 2]
    sample = completion(tokenizer(CALL), original, TOOLS)
    assert original == [1, 2]
    assert env.step(sample['action']) == (0., False)
    assistant, result = env.messages[-2:]
    assert assistant['role'] == 'assistant'
    assert assistant['tool_calls'][0]['function']['name'] == 'lookup'
    assert result['role'] == 'tool' and result['name'] == 'lookup'
    assert result['tool_call_id'] == assistant['tool_calls'][0]['id']
    assert json.loads(result['content']) == {'status': 200, 'tool_result': {'quantity': 7}}
    assert env.step(completion(tokenizer('7件'), [1, 2], TOOLS)['action']) == (0., True)
    assert env.messages[-1] == {'role': 'user', 'content': '收到'}


def test_multiple_calls_have_individual_results_and_do_not_submit_preface():
    env = NativeRuntime(None)
    env.reset(1)
    action = completion(tokenizer('查询中\n' + CALL + '\n' + CALL), [1, 2], TOOLS)['action']
    env.step(action)
    assistant, first, second = env.messages[-3:]
    assert assistant['content'] == '查询中'
    assert [first['tool_call_id'], second['tool_call_id']] == [c['id'] for c in assistant['tool_calls']]
    assert first['tool_call_id'] != second['tool_call_id']
    assert first['role'] == second['role'] == 'tool'
    assert not any(t['path'] == '/v1/agent_response' for t in env.trace)


@pytest.mark.parametrize('text', [
    '<tool_call>{"name":"lookup"', '<tool_call>bad json</tool_call>',
    '<tool_call>{"name":"lookup","arguments":[]}</tool_call>',
    CALL + '\n<tool_call>{', '<think>unfinished', '', '</tool_call>',
])
def test_invalid_or_truncated_calls_do_not_execute_or_submit(text):
    env = NativeRuntime(None)
    env.reset(1)
    before = len(env.trace)
    sample = completion(tokenizer(text), [1, 2], TOOLS)
    assert 'protocol_error' in sample['action']
    env.step(sample['action'])
    assert len(env.trace) == before
    assert 'protocol_error' in json.loads(env.messages[-1]['content'])
    assert 'protocol_error' not in env.messages[-2]
    assert not env.terminated


def test_unknown_tool_returns_correlated_tool_error():
    env = NativeRuntime(None)
    env.reset(1)
    before = len(env.trace)
    env.step(completion(tokenizer(CALL.replace('lookup', 'missing')), [1, 2], TOOLS)['action'])
    assert len(env.trace) == before
    assert env.messages[-1]['role'] == 'tool'
    assert env.messages[-1]['tool_call_id'] == env.messages[-2]['tool_calls'][0]['id']
    assert json.loads(env.messages[-1]['content'])['status'] == 400


def test_reasoning_is_separate_from_user_submission():
    env = NativeRuntime(None)
    env.reset(1)
    action = completion(tokenizer('<think>计算完成</think>\n7件'), [1, 2], TOOLS)['action']
    assert action['reasoning_content'] == '计算完成'
    env.step(action)
    assert env.conversation[-2]['content'] == '7件'


class NativePolicy:
    def encode(self, messages, *, tools):
        assert tools == TOOLS
        return [10]

    def sample(self, messages, max_tokens, max_context, *, tools, greedy=False):
        assert tools == TOOLS
        text = '7件' if messages[-1]['role'] == 'tool' else CALL
        return {'prompt': self.encode(messages, tools=tools), 'tokens': [1, 2], 'old_logp': [-.1, -.2],
                **completion(tokenizer(text), [1, 2], tools)}

    def iter_sample(self, messages, max_tokens, max_context, *, tools):
        yield None
        return self.sample(messages, max_tokens, max_context, tools=tools)


class NativeDecoder:
    def __init__(self, policy):
        self.policy, self.requests = policy, {}

    def add(self, key, messages, max_tokens, max_context, *, tools):
        self.requests[key] = (messages, max_tokens, max_context, tools)

    def tick(self):
        results = {key: self.policy.sample(messages, limit, context, tools=tools)
                   for key, (messages, limit, context, tools) in self.requests.items()}
        self.requests.clear()
        return results


class NativeBatchedPolicy(NativePolicy):
    def batch_decoder(self):
        return NativeDecoder(self)


@pytest.mark.parametrize('steps', [1, 2])
@pytest.mark.parametrize('mode', ['serial', 'parallel', 'batched'])
def test_rollout_paths_carry_tools_actions_and_bootstrap_context(steps, mode):
    config = {'rollout_group': 2, 'max_steps': steps, 'max_tokens': 16, 'max_context': 512,
              'algorithm': 'grpo', 'rollout_timeout': 10}
    policy = NativeBatchedPolicy() if mode == 'batched' else NativePolicy()
    events = []
    def observe(event, **fields):
        import copy
        events.append((event, copy.deepcopy(fields)))
    if mode == 'serial':
        results = [rollout(policy, NativeRuntime(None), config, 42, observe=observe)]
    else:
        pool = EnvironmentPool('unused', 2, factory=f'{__name__}:NativeRuntime')
        try:
            results = parallel_rollouts(policy, pool, config, 42, observe=observe)
        finally:
            pool.close()
    for result in results:
        assert result['terminated'] == (steps == 2)
        assert result['bootstrap_prompt'] == ([] if steps == 2 else [10])
        assert sum(m['role'] == 'tool' for m in result['messages']) == 1
        assert len(result['actions']) == steps
        assert result['actions'][0]['tokens'] == [1, 2]
        assert result['actions'][0]['old_logp'] == [-.1, -.2]
    for index in range(len(results)):
        episode_events = [(event, fields) for event, fields in events if fields['rollout_index'] == index]
        assert [event for event, _ in episode_events] == ['reset'] + ['action', 'step'] * steps + ['finish']
        assert episode_events[1][1]['sample']['action']['tool_calls'][0]['function']['name'] == 'lookup'
        assert episode_events[2][1]['messages'][-1]['role'] == 'tool'
