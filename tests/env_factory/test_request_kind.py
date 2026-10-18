"""Explicit world/task boundaries without guessing the kind of legacy artifacts."""
import copy
import json

import pytest

from env_factory.core import spec
from env_factory.production import world
from rl.data.tasks import load_tasks


def request():
    return {
        'id': 'world-00', 'family': 'retail', 'category': 'simple_agentic',
        'title': 'Sample', 'user_request': 'Help with my order', 'business_rules': 'Check the order.',
        'tables': {'orders': [{'id': 'O1', 'status': 'pending'}]},
        'user_facts': {'goals': [{'process': 'lookup', 'params': {'order_id': 'O1'}}]},
        'user_scene': {'situation': 'Shopping', 'goal': 'Check my order', 'known': {}, 'wants_to_know': []},
        'goal_schema': {'lookup': {'description': 'Look up an order', 'params': {'order_id': 'str'}}},
    }


def test_legacy_load_does_not_infer_kind_from_goal_schema(tmp_path):
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(request()))
    before = path.read_bytes()
    loaded = spec.load(path)
    assert 'request_kind' not in loaded
    assert 'world_id' not in loaded
    assert path.read_bytes() == before


@pytest.mark.parametrize('metadata', [
    {'request_kind': 'world'},
    {'request_kind': 'task'},
    {'request_kind': 'task', 'world_id': 'parent-world'},
])
def test_valid_kind(metadata):
    assert spec.validate({**request(), **metadata}) == []


@pytest.mark.parametrize('metadata', [
    {'request_kind': 'unknown'},
    {'request_kind': None},
    {'request_kind': 'world', 'world_id': 'parent'},
    {'world_id': 'parent'},
    {'request_kind': 'task', 'world_id': ''},
    {'request_kind': 'task', 'world_id': 12},
])
def test_invalid_kind_metadata(metadata):
    assert spec.validate({**request(), **metadata})


@pytest.mark.parametrize('marked', [False, True])
def test_composition_stamps_task_and_preserves_world(marked):
    parent = request()
    if marked:
        parent['request_kind'] = 'world'
    before = copy.deepcopy(parent)
    task = {
        'user_facts': {'goals': [{'process': 'lookup', 'params': {'order_id': 'O1'}}]},
        'user_scene': parent['user_scene'],
        'overlay': [{'table': 'orders', 'id': 'O1', 'set': {'status': 'delivered'}}],
    }
    result = world.compose(parent, task, 'world-00-t001')
    assert result['request_kind'] == 'task'
    assert result['world_id'] == parent['id']
    assert result['id'] == 'world-00-t001'
    assert result['goal_schema'] == parent['goal_schema']
    assert result['tables']['orders'][0]['status'] == 'delivered'
    assert 'user_request' not in result
    assert parent == before
    assert spec.validate({**result, 'user_request': 'Check this order'}) == []


def test_world_check_rejects_task_and_brief_labels_sample(tmp_path):
    value = {**request(), 'request_kind': 'task'}
    (tmp_path / 'request.json').write_text(json.dumps(value))
    problems, _ = world.check_world(tmp_path)
    assert 'not a task request' in problems[0]
    value['request_kind'] = 'world'
    value['time'] = {'start': 0, 'tick': 0}
    assert 'build-check sample' in spec.brief(value)


def test_rl_rejects_explicit_world_but_accepts_task_and_legacy(tmp_path, monkeypatch):
    from rl.environments import sandbox
    monkeypatch.setattr(sandbox, 'is_sandbox_bundle', lambda path: True)
    monkeypatch.setattr(sandbox, 'bundle_identity', lambda path: 'test-identity')
    path = tmp_path / 'request.json'
    path.write_text(json.dumps({**request(), 'request_kind': 'world'}))
    with pytest.raises(ValueError, match='world bundle is not a training task'):
        load_tasks(sandbox=str(tmp_path))
    for metadata in ({}, {'request_kind': 'task', 'world_id': 'parent-world'}):
        path.write_text(json.dumps({**request(), **metadata}))
        assert len(load_tasks(sandbox=str(tmp_path))) == 1
