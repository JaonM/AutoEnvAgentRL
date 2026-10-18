"""A partially admitted or changed dataset must never release training."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/bench'))
import world_sft_control as control


def write_release(path, mutate=None):
    rows = []
    for domain in ('retail', 'airline'):
        for split, n in (('train', 500), ('dev', 100)):
            rows.extend({'id': f'{domain}-{split}-{i}', 'domain': domain, 'split': split,
                         'scenario_family': split, 'success': True, 'policy_aware_success': True,
                         'quality_audit_passed': True, 'messages': [{'role': 'assistant', 'content': 'Done'}]}
                        for i in range(n))
    if mutate:
        mutate(rows)
    (path / 'teacher.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (path / 'valid_ids.json').write_text(json.dumps([r['id'] for r in rows if r['split'] == 'dev']))
    (path / 'tasks.json').write_text('{}')
    release = {'training_ready': True, 'train_tasks': 1000, 'dev_tasks': 200,
               'source_hashes': {name: control.digest(path / name)
                                for name in ('teacher.jsonl', 'valid_ids.json', 'tasks.json')}}
    (path / 'release.json').write_text(json.dumps(release))


def test_complete_release_passes(tmp_path):
    write_release(tmp_path)
    assert control.validate_release(tmp_path)['train_tasks'] == 1000


@pytest.mark.parametrize('change,reason', [
    (lambda rows: rows.pop(), 'task IDs'),
    (lambda rows: rows[0].update(id=rows[1]['id']), 'task IDs'),
    (lambda rows: rows[0].update(quality_audit_passed=False), 'unqualified'),
    (lambda rows: rows[0].update(scenario_family='dev'), 'overlap'),
    (lambda rows: rows[0].update(domain='airline'), 'quota'),
])
def test_fresh_hash_does_not_hide_bad_admission(tmp_path, change, reason):
    write_release(tmp_path, change)
    with pytest.raises(ValueError, match=reason):
        control.validate_release(tmp_path)


def test_file_change_after_release_rejected(tmp_path):
    write_release(tmp_path)
    with (tmp_path / 'teacher.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        control.validate_release(tmp_path)


def test_eval_refuses_to_overwrite_prior_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / 'output/tau2/qexisting-airline.log'
    evidence.parent.mkdir(parents=True)
    evidence.write_text('prior evidence')
    monkeypatch.setattr(control.subprocess, 'run', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='already exists'):
        control.evaluate('existing', Path('model'), 1, {}, Path('ids'), tmp_path, 9310, {})
    assert evidence.read_text() == 'prior evidence'


def test_as_ready_prioritizes_both_new_data_arms_before_second_seeds_and_old_controls():
    groups = [('world',), ('mixed',), ('old',)]
    assert [(g[0], s) for g, s in control.training_schedule(groups, True)] == [
        ('world', 0), ('mixed', 0), ('world', 1), ('mixed', 1), ('old', 0), ('old', 1)]
    assert [(g[0], s) for g, s in control.training_schedule(groups)] == [
        ('world', 0), ('world', 1), ('mixed', 0), ('mixed', 1), ('old', 0), ('old', 1)]


def test_partial_report_does_not_invent_pending_scores(tmp_path):
    control.write_results(tmp_path, {'world-s0': {'overall': {'tasks': 164, 'pass_at_1': .25},
                                               'write': {'pass_at_1': .2}}})
    text = (tmp_path / 'results.md').read_text()
    assert '25.00%' in text and '20.00%' in text
    assert '| world-s0 | 164 | 25.00% | — |' in text
    assert 'mixed-s0' not in text
