"""Opt-in original Qwen3-4B tests, separate from Instruct-2507 coverage."""
import json
import os
from pathlib import Path

import pytest

MODEL = os.environ.get('RL_TEST_QWEN3_MODEL')
pytestmark = pytest.mark.skipif(not MODEL, reason='set RL_TEST_QWEN3_MODEL to original Qwen3-4B MLX model')

TOOLS = [{'type': 'function', 'function': {'name': 'lookup',
    'description': '查询物品的准确数量。必须通过此工具获取数量，不能猜测。',
    'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}}}]


def save_evidence(name, value):
    if destination := os.environ.get('RL_VALIDATION_OUTPUT'):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


@pytest.mark.parametrize('mode', ['thinking', 'no-thinking'])
@pytest.mark.parametrize('decode', ['single', 'batch'])
def test_real_thinking_tool_roundtrip_and_behavior_scores(tmp_path, mode, decode):
    from rl.model import Policy
    import mlx.core as mx
    mx.random.seed(42)
    policy = Policy(MODEL, tuning='lora', thinking_mode=mode, critic=True)
    messages = [{'role': 'system', 'content': '你是工具助手。先调用 lookup 查询，不要猜测。获得结果后仅回复数量，格式为“数字件”。'},
                {'role': 'user', 'content': '请查询甲的数量。'}]
    prompt = policy.tokenizer.decode(policy.encode(messages, tools=TOOLS), skip_special_tokens=False)
    if mode == 'no-thinking':
        assert prompt.endswith('<think>\n\n</think>\n\n')
    else:
        assert prompt.endswith('<|im_start|>assistant\n')
    def sample(history):
        if decode == 'single':
            result = policy.sample(history, max_tokens=2048, max_context=4096, tools=TOOLS, greedy=True)
            result['validation_cached_max_logp_error'] = result['behavior_scoring_max_logp_error']
            return result
        decoder = policy.batch_decoder(greedy=True)
        for row in range(2):
            decoder.add(row, history, 2048, 4096, tools=TOOLS)
        results = {}
        while decoder.requests:
            results.update(decoder.tick())
        assert results[0]['tokens'] == results[1]['tokens']
        for result in results.values():
            # Batched decoding records its actual sampling distribution directly;
            # independently rescore here instead of expecting a single-path field.
            scores, _ = policy.cached_token_stats(mx.array([result['prompt'] + result['tokens']]), len(result['prompt']))
            error = float(mx.max(mx.abs(scores - mx.array(result['old_logp']))))
            result['validation_cached_max_logp_error'] = error
            assert error <= .001
        return results[0]
    call = sample(messages)
    assert call['generation_finish'] == 'eos', call['text']
    assert 'protocol_error' not in call['action'], call['text']
    calls = call['action'].get('tool_calls', [])
    assert len(calls) == 1 and calls[0]['function']['name'] == 'lookup', call['text']
    assert json.loads(calls[0]['function']['arguments']) == {'key': '甲'}
    if mode == 'thinking':
        assert '<think>' in call['text'] and '</think>' in call['text']
        assert call['action']['reasoning_content']
    else:
        assert not call['action'].get('reasoning_content')
    calls[0]['id'] = 'call_1'
    messages += [call['action'], {'role': 'tool', 'tool_call_id': 'call_1', 'name': 'lookup',
                                 'content': '{"quantity":7}'}]
    answer = sample(messages)
    assert answer['generation_finish'] == 'eos', answer['text']
    assert 'protocol_error' not in answer['action'], answer['text']
    assert answer['action']['content'].strip() == '7件', answer['text']
    for sample in (call, answer):
        assert sample['validation_cached_max_logp_error'] <= .001
        assert len(sample['old_logp']) == len(sample['old_values']) == len(sample['tokens'])
    artifact = {'mode': mode, 'decode': decode, 'prompt': prompt, 'call': call, 'answer': answer}
    (tmp_path / f'{mode}.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    save_evidence(f'tool-roundtrip-{decode}-{mode}.json', artifact)


@pytest.mark.parametrize('mode', ['thinking', 'no-thinking'])
@pytest.mark.parametrize('algorithm', ['ppo', 'grpo'])
def test_original_qwen3_training_modes(tmp_path, monkeypatch, mode, algorithm):
    from test_training_batches_integration import test_training_minibatches_and_critic_checkpoint as run
    run(tmp_path, monkeypatch, algorithm, model_path=MODEL, thinking_mode=mode)
    save_evidence(f'training-{algorithm}-{mode}.json', {
        'algorithm': algorithm, 'mode': mode, 'tuning': 'lora',
        'scope': 'real model updates and checkpoint resume; controlled rewards and stubbed evaluation',
        'optimizer_metrics': json.loads((tmp_path / algorithm / 'optimizer_metrics.json').read_text())})


@pytest.mark.parametrize('mode', ['thinking', 'no-thinking'])
def test_qat_update_export_preserves_logits(tmp_path, mode):
    import mlx.core as mx
    import mlx.optimizers as optim
    from rl.model import Policy
    from rl.actor import ActorTrainer
    from rl.train import Config
    from rl.timing import StageTimes
    mx.random.seed(42)
    policy = Policy(MODEL, tuning='qat', thinking_mode=mode)
    reference = Policy(MODEL, tuning='qat', thinking_mode=mode)
    initial = policy.save_snapshot(tmp_path / 'snapshots', 0)
    reference.restore(initial)
    reference.freeze()
    config = Config(thinking_mode=mode, learning_rate=1e-5, target_kl=0., numerical_check_mode='strict')
    actor = ActorTrainer(policy, reference, optim.Adam(config.learning_rate), config, StageTimes())
    decoder = policy.batch_decoder()
    for index in range(2):
        decoder.add(index, [{'role': 'user', 'content': 'Invent a short name for a new animal.'}], 32, 4096)
    samples = {}
    while decoder.requests:
        samples.update(decoder.tick())
    rows = [{**sample, 'advantage': 1. if index else -1., 'weight': .5} for index, sample in sorted(samples.items())]
    before = policy.digest(policy_only=True)
    result = actor.step(rows)
    assert 'skipped' not in result and policy.digest(policy_only=True) != before
    description = policy.export_qat(tmp_path)
    assert description
    ids = mx.array([rows[0]['prompt'] + rows[0]['tokens']])
    expected = policy.token_stats(ids, len(rows[0]['prompt']))[0]
    mx.eval(expected)
    policy.load_qat_export(tmp_path)
    error = float(mx.max(mx.abs(policy.token_stats(ids, len(rows[0]['prompt']))[0] - expected)))
    assert error < .01
    save_evidence(f'qat-{mode}.json', {'mode': mode, 'tuning': 'qat', 'update': result,
        'packed_reload_max_logp_error': error, 'scope': 'real model update with controlled advantages'})
