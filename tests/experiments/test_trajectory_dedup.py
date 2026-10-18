"""Task-local dedup must remove repetitions without discarding distinct decisions."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from trajectory_dedup import deduplicate, task_key, validation_keys
from test_sft_signal import episode
import sft_train
from sample_world_tasks import select_trajectories


def test_rephrasing_call_ids_json_format_and_quality_representative():
    a = episode()
    b = copy.deepcopy(a)
    a['reward'], b['reward'] = .9, 1.0
    b['seed'] = 1
    b['messages'][-1]['content'] = 'Your pack is now activated.'
    for m in b['messages']:
        for c in m.get('tool_calls') or []:
            c['id'] += '_new'
            c['function']['arguments'] = '{ "account_id" : "acct_1" }'
        if m.get('role') == 'tool':
            m['tool_call_id'] += '_new'
            m['content'] = json.dumps(json.loads(m['content']), sort_keys=True)
    original = copy.deepcopy([a, b])
    kept, audit = deduplicate([a, b])
    assert kept == [b]
    assert audit['dropped'] == 1
    assert audit['comparisons'][0]['kept']['seed'] == 1
    assert deduplicate([b, a])[0] == [b]
    assert [a, b] == original


@pytest.mark.parametrize('change', ['task', 'domain', 'world', 'consent', 'argument', 'result', 'recovery',
                                  'clarification', 'number', 'negation', 'policy', 'outcome'])
def test_distinct_paths_survive(change):
    a = episode()
    b = copy.deepcopy(a)
    if change in ('task', 'domain', 'world', 'consent'):
        b['id' if change == 'task' else change] = 'different'
    elif change == 'argument':
        b['messages'][2]['tool_calls'][0]['function']['arguments'] = '{"account_id":"acct_2"}'
    elif change == 'result':
        b['messages'][5]['content'] = '{"ok":true,"result":{"balance":2001}}'
    elif change == 'recovery':
        b['messages'] = b['messages'][:2] + b['messages'][4:]
    elif change == 'clarification':
        b['messages'][2:2] = [{'role': 'assistant', 'content': 'Which account?'},
                             {'role': 'user', 'content': 'acct_1'}]
    elif change == 'number':
        a['messages'][-1]['content'] = 'Your refund is now processed: 123.45 dollars.'
        b['messages'][-1]['content'] = 'Your refund is now processed: 123.46 dollars.'
    elif change == 'negation':
        b['messages'][-1]['content'] = 'Your pack is not activated.'
    elif change == 'policy':
        b['messages'][0]['content'] += ' Do not modify the account.'
    elif change == 'outcome':
        b.update(success=False, near_miss=True)
    assert len(deduplicate([a, b])[0]) == 2


def test_direct_answers_are_not_collapsed_without_text_match():
    a = episode()
    a['messages'] = a['messages'][:2] + [a['messages'][-1]]
    b = copy.deepcopy(a)
    b['messages'][-1]['content'] = 'Contact the service desk to discuss other options.'
    assert len(deduplicate([a, b])[0]) == 2
    b['messages'][-1]['content'] = 'Your pack is now activated.'
    assert len(deduplicate([a, b])[0]) == 1


def test_platform_aliases_and_argument_types():
    a = episode()
    a['messages'][2]['tool_calls'][0]['function'] = {'name': 'v3_ask', 'arguments': '{"consent":"123"}'}
    b = copy.deepcopy(a)
    b['naming'] = {'v3_ask': 'query_customer', 'consent': 'confirmation_token', '_text': 1}
    b['messages'][2]['tool_calls'][0]['function'] = {'name': 'query_customer', 'arguments': '{"confirmation_token":"123"}'}
    assert len(deduplicate([a, b])[0]) == 1
    b['messages'][2]['tool_calls'][0]['function']['arguments'] = '{"confirmation_token":123}'
    assert len(deduplicate([a, b])[0]) == 2


def test_business_values_are_not_rewritten_as_tool_aliases():
    a = episode()
    a['naming'] = {'v3_ask': 'query_customer'}
    b = copy.deepcopy(a)
    a['messages'][2]['tool_calls'][0]['function']['arguments'] = '{"account_id":"v3_ask"}'
    b['messages'][2]['tool_calls'][0]['function']['arguments'] = '{"account_id":"query_customer"}'
    assert len(deduplicate([a, b])[0]) == 2


def test_batch_and_mixed_speech_preserved_and_result_linkage_checked():
    a = episode()
    a['messages'][2]['tool_calls'].append(copy.deepcopy(a['messages'][4]['tool_calls'][0]))
    a['messages'][2]['content'] = 'I will check this for you.'
    del a['messages'][4]
    kept, _ = deduplicate([a, copy.deepcopy(a)])
    assert kept == [a]
    assert len(kept[0]['messages'][2]['tool_calls']) == 2
    assert kept[0]['messages'][2]['content']
    b = copy.deepcopy(a)
    b['messages'][3]['tool_call_id'], b['messages'][4]['tool_call_id'] = 'b', 'a'
    assert len(deduplicate([a, b])[0]) == 2


def test_missing_ids_are_not_compared_and_disable_is_explicit():
    a = episode()
    del a['id']
    assert len(deduplicate([a, copy.deepcopy(a)])[0]) == 2
    a['id'] = 'a'
    assert len(deduplicate([a, copy.deepcopy(a)], enabled=False)[0]) == 2
    with pytest.raises(ValueError):
        deduplicate([a], threshold=0)


def test_dedup_runs_before_augmentation_and_replay(tmp_path, monkeypatch):
    a = episode()
    path = tmp_path / 'teacher.jsonl'
    path.write_text(json.dumps(a) + '\n')
    calls = []
    import tool_rename
    monkeypatch.setattr(tool_rename, 'augment', lambda e, seed: calls.append(seed) or e)
    args = SimpleNamespace(teacher=str(path), replay=[str(path)], seed=0, split_seed=17, augmentation_seed=31,
                           rename_tools=1, relayout=0, greet=0, example_values=0, inject_errors=0,
                           vary_results=0, context_only_announcements=False, context_only_reports=0)
    assert len(sft_train.prepare_episodes(args)) == len(calls) == 1
    assert args.dedup_report['dropped'] == 1


def test_all_variants_of_task_share_split():
    a = episode()
    b = {**episode(), 'id': 'task-2'}
    c = {**episode(), 'seed': 1, 'world': 'flip:approval'}
    held = validation_keys([a, b, c], .5)
    assert task_key(a) in held and task_key(c) in held and task_key(b) not in held


def test_pilot_selection_never_changes_evaluation_or_admits_diagnostics():
    a = {**episode(), 'pilot_accepted': True}
    record = {'domain': 'test', 'index': 1, 'task_certified': True, 'raw_episodes': [a, copy.deepcopy(a)],
              'raw_successes': 2, 'world': 'original', 'consent': 'plain'}
    original = copy.deepcopy(record)
    report = select_trajectories(record)
    assert report['input'] == 2 and report['kept'] == 1
    assert report['training_ready'] is False
    assert len(report['selected_sources']) == 1
    assert record == original
    record['diagnostic_only'] = True
    assert select_trajectories(record)['kept'] == 0
    record['diagnostic_only'], record['task_certified'] = False, False
    assert select_trajectories(record)['kept'] == 0
    del record['task_certified']
    assert select_trajectories(record)['kept'] == 0
    assert select_trajectories(record, certified=True)['kept'] == 1


def test_cli_writes_audit_and_preserves_raw(tmp_path):
    a = episode()
    source, output = tmp_path / 'teacher.jsonl', tmp_path / 'selected.jsonl'
    source.write_text(''.join(json.dumps(e) + '\n' for e in [a, a, {**a, 'success': False}]))
    original = source.read_bytes()
    command = [sys.executable, 'experiments/trajectory_dedup.py', '--teacher', str(source), '--output', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert len(output.read_text().splitlines()) == 1
    report = json.loads(output.with_suffix('.dedup.json').read_text())
    assert report['dropped'] == 1 and report['ineligible'] == 1
    assert source.read_bytes() == original
    assert subprocess.run(command[:-1] + [str(source)], capture_output=True).returncode != 0


def test_selector_and_exporter_apply_same_filter(tmp_path):
    a = episode()
    b = {**episode(), 'seed': 2, 'world': 'flip:approval'}
    c = {**episode(), 'id': 'task-2'}
    source = tmp_path / 'teacher.jsonl'
    source.write_text(''.join(json.dumps(e) + '\n' for e in [a, a, b, c]))
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()
    (sandbox / 'request.json').write_text(json.dumps({'category': 'tool_use', 'language': 'en'}))
    tasks = tmp_path / 'tasks.json'
    tasks.write_text(json.dumps({'tasks': [{'id': key, 'sandbox': str(sandbox)} for key in ('task-1', 'task-2')]}))
    selected = tmp_path / 'selected.jsonl'
    subprocess.run([sys.executable, 'experiments/sft_select.py', '--teacher', str(source), '--tasks', str(tasks),
                    '--output', str(selected), '--dev-domains', '0', '--domain-share', '1', '--allow-short'],
                   check=True, capture_output=True)
    assert len(selected.read_text().splitlines()) == 3
    assert json.loads(selected.with_suffix('.dedup.json').read_text())['dropped'] == 1
    out = tmp_path / 'export'
    subprocess.run([sys.executable, 'experiments/sft_data.py', '--teacher', str(source), '--output', str(out)],
                   check=True, capture_output=True)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['train']['episodes'] + summary['valid']['episodes'] == 3
    assert sorted([summary['train']['episodes'], summary['valid']['episodes']]) == [1, 2]
    assert json.loads((out / 'dedup_report.json').read_text())['dropped'] == 1
