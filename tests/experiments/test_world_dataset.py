"""Long-running generation must never release undersized or duplicated task sets."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
dataset = importlib.import_module('generate_world_dataset')


def setup_run(tmp_path, monkeypatch, *, accepted=True, duplicate=False):
    domains = []
    for split in ('train', 'dev'):
        world = tmp_path / f'world-{split}'
        world.mkdir()
        (world / 'manifest.json').write_text('{}')
        domains.append({'name': split, 'business_domain': 'library', 'split': split, 'bundle': str(world),
                        'assignments': [{'slot': 0, 'index': 0, 'scenario_family': split, 'template_id': split}],
                        'data_scope': {'owners': [], 'private_row_ids': {}, 'ownership': {'tables': {}}, 'references': []}})
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'domains': domains, 'targets': {'train': 1, 'dev': 1}}))
    readiness = tmp_path / 'ready.json'
    readiness.write_text(json.dumps({'completed': 1, 'total': 1, 'tasks_with_candidate': 1}))
    output = tmp_path / 'dataset'
    monkeypatch.setattr(sys, 'argv', ['run', '--plan', str(plan), '--output', str(output),
                                     '--readiness', str(readiness), '--task-attempts', '2'])
    monkeypatch.setattr(dataset, 'load_env_file', lambda: None)
    monkeypatch.setattr(dataset, 'ChatClient', lambda role: SimpleNamespace(model='test', base_url='http://test'))
    monkeypatch.setattr(dataset, 'CodexRunner', lambda: SimpleNamespace(model='test-author'))
    monkeypatch.setattr(dataset.bundle, 'load_bundle', lambda path, **kw: {
        'request': {'tables': {}}, 'source': str(path)})
    monkeypatch.setattr(dataset, 'draft_batch', lambda d, a, out, runner: out)
    monkeypatch.setattr(dataset, 'certify_task', lambda d, a, folder, out, runner: {
        'passed': True, 'stage': 'gates', 'bundle': f"{d['name']}/{a['index']}"})
    monkeypatch.setattr(dataset, 'fingerprint', lambda loaded: 'same' if duplicate else loaded['source'])
    attempts = []
    def collect(d, a, row, out, budget):
        attempts.append((d['name'], a['index']))
        path = out / f"{d['name']}-{a['index']}.json"
        episode = {'id': d['name'], 'domain': 'library', 'split': d['split'], 'scenario_family': d['split'],
                   'sandbox': row['bundle'], 'success': True}
        path.write_text(json.dumps(episode))
        return {'passed': accepted, 'selected_episode': str(path)}
    monkeypatch.setattr(dataset, 'collect', collect)
    return output, attempts


def test_release_requires_every_admitted_task_and_disjoint_split(tmp_path, monkeypatch):
    output, attempts = setup_run(tmp_path, monkeypatch)
    dataset.main()
    release = json.loads((output / 'release.json').read_text())
    assert release['training_ready']
    assert release['train_tasks'] == release['dev_tasks'] == 1
    assert len(attempts) == 2
    assert json.loads((output / 'valid_ids.json').read_text()) == ['dev']


def test_failure_replenishes_same_slots_with_new_indices_then_stops_without_release(tmp_path, monkeypatch):
    output, attempts = setup_run(tmp_path, monkeypatch, accepted=False)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        dataset.main()
    assert set(attempts) == {('train', 0), ('dev', 0), ('train', 1), ('dev', 1)}
    assert not (output / 'release.json').exists()
    assert not json.loads((output / 'progress.json').read_text())['training_ready']


def test_duplicate_business_tasks_cannot_fill_two_slots(tmp_path, monkeypatch):
    output, attempts = setup_run(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        dataset.main()
    assert len(attempts) == 1
    assert not (output / 'release.json').exists()
