"""Finalization preserves frozen intent; native review outcomes are protocol mocks."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
import finalize_v2_free_assets as finalizer
from env_factory_v2.demand import layered_demand


def draft_assets(root, count):
    root.mkdir()
    samples = []
    strata = ['direct', 'simple', 'ordinary', 'complex', 'complex'] * (count // 5)
    for identity, stratum in enumerate(strata, 1):
        folder = root / f'{identity:02d}'
        folder.mkdir()
        values = {'business': {'policy': 'Fixture policy', 'tables': []},
                  'user': {'facts': {}}, 'acceptance': {},
                  'user_scene': {'desired_outcome': f'unchanged outcome {identity}',
                                 'known_information': {'value': identity}}}
        for name, value in values.items():
            (folder / f'{name}.json').write_text(json.dumps(value))
        request = f'original request {identity}'
        samples.append({'id': identity, 'stratum': stratum, 'raw_request': request,
            **{key: f'{identity:02d}/{key}.json' for key in ('business', 'user', 'acceptance')},
            'demand': layered_demand(request, 'Fixture policy', document_id=f'{identity:02d}/business.json',
                                    graph_evidence={'backend': 'fixture'})})
    (root / 'manifest.json').write_text(json.dumps({'samples': samples, 'mix': dict(Counter(strata))}))
    seal(root)


def seal(root):
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*.json') if p.name != 'frozen.json'}
    (root / 'frozen.json').write_text(json.dumps({'files': files}))


@pytest.mark.parametrize('count,repair', [(20, ''), (40, ''), (40, '3,16')])
def test_finalization_scales_without_sample_specific_scene_edits(tmp_path, monkeypatch, count, repair):
    draft, output = tmp_path / 'draft', tmp_path / 'output'
    draft_assets(draft, count)
    original = {str(p.relative_to(draft)): p.read_bytes() for p in draft.rglob('*.json')}
    generated, reviewed, scopes = [], [], []

    class Agent:
        def ask(self, prompt, payload, schema):
            if 'samples' in payload:
                reviewed.extend(payload['samples'])
                return {'checks': [{'id': row['id'], 'passed': True, 'reason': 'mock'}
                                   for row in payload['samples']]}, {'mock': True}
            generated.append(payload)
            return {'user_request': 'rewritten ' + payload['desired_outcome']}, {'mock': True}

    def scope(agent, **kwargs):
        scopes.append(kwargs['user_request'])
        return {'passed': True, 'answer_obligations': {'mock': True}, 'generation_output_scope': {'mock': True}}

    monkeypatch.setattr(finalizer.NativeCodeAgent, 'from_connection', lambda *a, **k: Agent())
    monkeypatch.setattr('audit_v2_natural.canonical_truth', lambda *a: {})
    monkeypatch.setattr('v2_answer_obligations.review_case_answer_obligations', scope)
    monkeypatch.setattr(sys, 'argv', ['finalize', '--draft', str(draft), '--output', str(output),
                                     '--connection', 'unused', '--repair-ids', repair])
    finalizer.main()
    result = finalizer.verify_assets(output)
    assert len(result['samples']) == len(reviewed) == len(scopes) == count
    assert len(generated) == (2 if repair else 0)
    for path, contents in original.items():
        assert (draft / path).read_bytes() == contents
        if path.endswith('user_scene.json'):
            assert (output / path).read_bytes() == contents
    for row in result['samples']:
        expected = (f"rewritten unchanged outcome {row['id']}" if str(row['id']) in repair.split(',')
                    else f"original request {row['id']}")
        assert row['raw_request'] == expected


@pytest.mark.parametrize('checks', [
    [{'id': 1, 'passed': True}, {'id': 1, 'passed': True}],
    [{'id': 1, 'passed': True}],
    [{'id': True, 'passed': True}, {'id': 2, 'passed': True}],
    [{'id': 1, 'passed': 1}, {'id': 2, 'passed': True}],
    [{'id': 1, 'passed': True}, {'id': 2, 'passed': False}],
])
def test_review_cannot_omit_duplicate_or_coerce_pass(checks):
    with pytest.raises(ValueError):
        finalizer.validate_review({'checks': checks}, {1, 2})


def test_unsealed_scene_is_rejected_before_copy_or_model_call(tmp_path, monkeypatch):
    draft, output = tmp_path / 'draft', tmp_path / 'output'
    draft_assets(draft, 20)
    frozen = json.loads((draft / 'frozen.json').read_text())
    del frozen['files']['03/user_scene.json']
    (draft / 'frozen.json').write_text(json.dumps(frozen))
    monkeypatch.setattr(sys, 'argv', ['finalize', '--draft', str(draft), '--output', str(output),
                                     '--connection', 'unused'])
    with pytest.raises(ValueError, match='Every user scene must be frozen'):
        finalizer.main()
    assert not output.exists()
