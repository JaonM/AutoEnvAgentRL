"""Protocol checks only: reused synthetic fixtures and mocked reviews are not unseen yield."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
import prepare_v2_cross_openings as openings
import freeze_v2_cross_business as freeze
from v2_preparation_pack import load_pack, identity, exact_ids, user_scenes, business_review_input


def pack_file(tmp_path, count=40):
    pack = load_pack()
    if count == 40:
        extra = copy.deepcopy(pack['cases'])
        for row in extra:
            row['id'] += 100  # sparse IDs must work, not just range(1, 41)
        pack['cases'] += extra
    path = tmp_path / 'pack.json'
    path.write_text(json.dumps(pack))
    return path, pack


def review_evidence(tmp_path, pack):
    roots = [tmp_path / name for name in ('generated', 'natural', 'business')]
    for root in roots:
        root.mkdir()
        save(root, 'preparation_identity.json', identity(pack))
    requests = [{'id': row['id'], 'user_request': f"mock opening {row['id']}"} for row in pack['cases']]
    save(roots[0], 'generated_openings.json', {'response': {'openings': requests}})
    save(roots[1], 'review_input.json', {'scenes': user_scenes(pack), 'openings': requests})
    save(roots[2], 'review_input.json', business_review_input(pack, requests))
    for root, file, flags in [(roots[1], 'independent_review.json', ['natural', 'faithful', 'no_invented_facts_or_steps']),
                             (roots[2], 'business_review.json', ['genuine_rule_or_composition_novelty',
                                                              'public_demand_consistent', 'public_policy_sufficient'])]:
        save(root, file, {'response': {'reviews': [{'id': row['id'], **dict.fromkeys(flags, True)}
                                                  for row in pack['cases']]}})
    return roots


def save(root, name, value):
    (root / name).write_text(json.dumps(value))


@pytest.mark.parametrize('count', [20, 40])
def test_scaled_pack_review_preserves_inputs_and_evidence_identity(tmp_path, count):
    path, pack = pack_file(tmp_path, count)
    roots = review_evidence(tmp_path, pack)
    rows, evidence = freeze.reviewed_cases(*roots, path)
    assert len(rows) == count
    assert evidence['preparation_identity'] == identity(pack)
    assert evidence['preparation_pack'] == pack
    assert rows[0]['raw_request'] == 'mock opening 1'
    assert json.loads(path.read_text()) == pack


@pytest.mark.parametrize('field', ['user', 'acceptance', 'user_scene', 'historical_business_rules'])
def test_changed_inputs_cannot_reuse_passed_reviews(tmp_path, field):
    path, pack = pack_file(tmp_path)
    roots = review_evidence(tmp_path, pack)
    if field == 'historical_business_rules':
        pack[field][0]['policy'] += ' changed'
    elif field == 'user':
        pack['cases'][0][field]['facts']['new'] = 'changed'
    elif field == 'user_scene':
        pack['cases'][0][field]['known_information']['new'] = 'changed'
    else:
        pack['cases'][0][field]['params']['new'] = 'changed'
    path.write_text(json.dumps(pack))
    with pytest.raises(ValueError, match='differs from reviewed evidence'):
        freeze.reviewed_cases(*roots, path)


def test_business_review_must_really_receive_private_facts_and_relations(tmp_path):
    path, pack = pack_file(tmp_path)
    roots = review_evidence(tmp_path, pack)
    review = json.loads((roots[2] / 'review_input.json').read_text())
    review['new_cases'][1]['clarifiable_private_user_information'] = {}
    save(roots[2], 'review_input.json', review)
    with pytest.raises(ValueError, match='assets differ'):
        freeze.reviewed_cases(*roots, path)


def test_unknown_scene_content_cannot_leak_acceptance_to_opening_author(tmp_path):
    path, pack = pack_file(tmp_path)
    pack['cases'][0]['user_scene']['acceptance'] = {'answer': 42}
    path.write_text(json.dumps(pack))
    with pytest.raises(ValueError, match='User scene'):
        load_pack(path)


@pytest.mark.parametrize('ids', [[1, 1], [True, 2], [0, 2], [1, 3]])
def test_identity_coverage_is_exact(ids):
    with pytest.raises(ValueError):
        exact_ids([{'id': i} for i in ids], {1, 2})


def test_generation_uses_only_supplied_scenes_for_sparse_forty_pack(tmp_path, monkeypatch):
    path, pack = pack_file(tmp_path)
    captured = []
    class Agent:
        def ask(self, prompt, payload, schema):
            captured.append(payload)
            return {'openings': [{'id': row['id'], 'user_request': 'mock'} for row in payload['scenes']]}, {}
    monkeypatch.setattr(openings.NativeCodeAgent, 'from_connection', lambda *a, **k: Agent())
    output = tmp_path / 'generated'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--case-pack', str(path), '--connection', 'unused',
                                     '--output', str(output)])
    openings.main()
    assert captured == [{'scenes': user_scenes(pack)}]
    assert json.loads((output / 'preparation_identity.json').read_text()) == identity(pack)


def test_forty_case_freeze_is_accepted_by_runner_with_same_graph_evidence(tmp_path, monkeypatch):
    from env_factory_v2.expressions import canonical
    from measure_v2_natural import verify_assets
    path, pack = pack_file(tmp_path)
    roots = review_evidence(tmp_path, pack)
    monkeypatch.setattr(freeze.NativeCodeAgent, 'from_connection', lambda *a, **k: object())
    monkeypatch.setattr(freeze, 'canonical_truth', lambda *a: {})
    monkeypatch.setattr(freeze, 'review_answer_obligations', lambda *a, **k:
                        {'passed': True, 'answer_obligations': {'mock': True}})
    documents = []
    class Driver:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(freeze.GraphDatabase, 'driver', lambda *a, **k: Driver())
    monkeypatch.setattr(freeze, 'install', lambda driver, namespace, rows, **k: documents.extend(rows))
    monkeypatch.setattr(freeze, 'retrieve', lambda *a, **k: {'documents': [
        {'document_id': row['document_id'], 'sha256': hashlib.sha256(canonical(row['business']).encode()).hexdigest(),
         'policy_summary': row['business']['policy'], 'relations': row['business']['relations']} for row in documents]})
    output = tmp_path / 'frozen'
    monkeypatch.setattr(sys, 'argv', ['freeze', '--case-pack', str(path), '--connection', 'unused',
        '--openings', str(roots[0]), '--opening-review', str(roots[1]), '--business-review', str(roots[2]),
        '--output', str(output), '--namespace', 'test-only'])
    freeze.main()
    manifest = verify_assets(output)
    assert len(manifest['samples']) == 40
    assert manifest['mix'] == {'direct': 8, 'simple': 8, 'ordinary': 8, 'complex': 16}
    evidence = json.loads((output / 'preparation_review.json').read_text())
    assert evidence['preparation_pack'] == pack
    assert evidence['preparation_identity'] == identity(pack)
