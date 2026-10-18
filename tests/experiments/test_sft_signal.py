"""Regressions for failure masking, successful recovery and controlled SFT seeds."""
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
import sft_train
import tool_rename


def episode():
    def call(i, name, args):
        return {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': i, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
    def result(i, value):
        return {'role': 'tool', 'tool_call_id': i, 'content': json.dumps(value)}
    return {'id': 'task-1', 'seed': 0, 'success': True, 'tools': [
        {'type': 'function', 'function': {'name': n, 'description': n, 'parameters': {
            'type': 'object', 'properties': {'account_id': {'type': 'string'}}, 'required': ['account_id']}}}
        for n in ('activate_pack', 'get_credit')], 'messages': [
            {'role': 'system', 'content': 'Help the customer.'},
            {'role': 'user', 'content': 'Activate my pack on acct_1.'},
            call('a', 'activate_pack', {'account_id': 'acct_1'}),
            result('a', {'ok': False, 'error': 'reviews required'}),
            call('b', 'get_credit', {'account_id': 'acct_1'}),
            result('b', {'ok': True, 'result': {'balance': 2000}}),
            call('c', 'activate_pack', {'account_id': 'acct_1'}),
            result('c', {'ok': True, 'result': 'PACK_ACTIVATED'}),
            {'role': 'assistant', 'content': 'Your pack is activated.'}]}


@pytest.mark.parametrize('style', ['envelope', 'raw', 'text'])
def test_error_label_survives_result_presentation(style):
    seed = next(str(i) for i in range(100) if random.Random(f'{i}-envelope').choice(
        ['envelope', 'raw', 'raw', 'text']) == style)
    before = episode()
    after = tool_rename.vary_results(before, seed)
    assert sft_train.failed_turns(after['messages']) == {2}
    assert after['messages'][3]['tool_failure'] is True
    assert after['messages'][7]['tool_failure'] is False
    assert 'tool_failure' not in before['messages'][3]
    if style == 'text':
        assert after['messages'][3]['content'] == 'reviews required'


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    path = Path('models/Qwen3-0.6B')
    if not path.exists():
        pytest.skip('local Qwen3 tokenizer required for rendering regression')
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.parametrize('mode', ['episode', 'turn'])
def test_loss_excludes_failure_and_keeps_successful_identical_retry(tokenizer, mode):
    e = episode()
    seed = next(str(i) for i in range(100) if random.Random(f'{i}-envelope').choice(
        ['envelope', 'raw', 'raw', 'text']) == 'text')
    e = tool_rename.vary_results(e, seed)
    def count(value):
        pairs = sft_train.render_episode(tokenizer, value, True) if mode == 'episode' else \
            sft_train.render_turns(tokenizer, value, False, 1500, skip_failed=True)
        return sum(sum(mask) for _, mask in pairs)
    baseline = count(e)
    failure_hidden = copy.deepcopy(e)
    failure_hidden['messages'][2]['context_only'] = True
    assert count(failure_hidden) == baseline  # already excluded by preserved failure label
    success_hidden = copy.deepcopy(e)
    success_hidden['messages'][6]['context_only'] = True
    assert count(success_hidden) < baseline  # successful identical retry must train
    pairs = sft_train.render_episode(tokenizer, e, True)
    assert 'tool_failure' not in tokenizer.decode(pairs[0][0])


def test_fixed_data_seeds_ignore_training_seed(tmp_path):
    path = tmp_path / 'teacher.jsonl'
    values = []
    for i in range(16):
        e = episode(); e['id'] = f'task-{i}'; values.append(e)
    path.write_text(''.join(json.dumps(e) + '\n' for e in values))
    args = SimpleNamespace(teacher=str(path), replay=[], seed=0, split_seed=17, augmentation_seed=31,
                           rename_tools=.7, relayout=.5, greet=.5, example_values=.5, inject_errors=.5,
                           vary_results=.6, context_only_announcements=False, context_only_reports=0)
    first = sft_train.prepare_episodes(args)
    args.seed = 1
    second = sft_train.prepare_episodes(args)
    assert first == second
    args.augmentation_seed = 32
    assert sft_train.prepare_episodes(args) != second


def test_public_mix_keeps_world_augmentation_and_validation_identical(tmp_path, tokenizer):
    teacher, public, held = [tmp_path / n for n in ('teacher.jsonl', 'public.jsonl', 'valid.json')]
    values = [{**episode(), 'id': f'world-{i}'} for i in range(12)]
    teacher.write_text(''.join(json.dumps(e) + '\n' for e in values))
    held.write_text(json.dumps(['world-0', 'world-1']))
    original = {**episode(), 'id': 'public:x', 'public': True}
    public.write_text(json.dumps(original) + '\n')
    args = SimpleNamespace(teacher=str(teacher), replay=[], valid_ids=str(held), seed=0,
                           split_seed=0, augmentation_seed=0, rename_tools=.7, relayout=.5,
                           greet=.5, example_values=.5, inject_errors=.5, vary_results=.6,
                           context_only_announcements=True, context_only_reports=0)
    baseline = sft_train.prepare_episodes(args)
    args.public_replay = [str(public)]
    mixed = sft_train.prepare_episodes(args)
    assert mixed[:-1] == baseline
    assert mixed[-1] == original
    args.seed = 1
    assert sft_train.prepare_episodes(args) == mixed
    ids = set(json.loads(held.read_text()))
    render_valid = lambda es: [sft_train.render_episode(tokenizer, e, True) for e in es if e['id'] in ids]
    assert render_valid(mixed) == render_valid(baseline)
    assert args.dedup_report['input'] == args.dedup_report['kept'] == 13
    # Public data cannot silently acquire a held-out world task's identity.
    public.write_text(json.dumps({**original, 'id': 'world-0'}) + '\n')
    with pytest.raises(ValueError, match='disjoint'):
        sft_train.prepare_episodes(args)


def test_policy_rejection_cannot_reenter_training_as_a_near_miss(tmp_path):
    valid = episode()
    rejected = {**episode(), 'id': 'rejected', 'success': False, 'near_miss': True,
                'state_ok': True, 'safety': True, 'reward': 1.0,
                'trajectory_policy': {'passed': False}}
    stale_success = {**rejected, 'id': 'stale-success', 'success': True}
    path = tmp_path / 'teacher.jsonl'
    path.write_text(''.join(json.dumps(e) + '\n' for e in [valid, rejected, stale_success]))
    args = SimpleNamespace(teacher=str(path), replay=[], seed=0, split_seed=17, augmentation_seed=31,
                           rename_tools=0, relayout=0, greet=0, example_values=0, inject_errors=0,
                           vary_results=0, context_only_announcements=False, context_only_reports=0)
    assert [e['id'] for e in sft_train.prepare_episodes(args)] == ['task-1']
