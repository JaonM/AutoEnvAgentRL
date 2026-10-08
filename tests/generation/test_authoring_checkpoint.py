import json

import pytest

from env_factory.generation.authoring_checkpoint import load_checkpoint, save_checkpoint
from tests.generation.test_business_blueprint import blueprint, request, implementation


def save(root, source=None):
    return save_checkpoint(root, request=request(), blueprint=blueprint(), source=source,
        evidence={'status': 'failed', 'author_seconds': 600, 'attempts': [
            {'phase': 'business_blueprint'}, {'phase': 'implementation', 'issue_ids': ['invalid-count']}],
            'repair_budget': {'used': {'structural': 1, 'semantic': 0}}})


def test_fixed_design_roundtrip_preserves_retry_evidence_without_certifying(tmp_path):
    path = save(tmp_path, implementation())
    loaded = load_checkpoint(path, expected_request=request())
    assert loaded['source'] == implementation()
    assert loaded['blueprint'] == blueprint()
    assert loaded['manifest']['source_attempts'] == 1
    assert loaded['manifest']['repair_budget']['used']['structural'] == 1
    assert loaded['manifest']['previous_issue_ids'] == ['invalid-count']
    assert loaded['manifest']['scope'] == 'candidate_only_requires_fresh_validation'
    with pytest.raises(ValueError, match='request differs'):
        load_checkpoint(path, expected_request=dict(request(), user_script_count=2))


def test_publishing_new_candidate_preserves_old_content_addressed_snapshot(tmp_path):
    path = save(tmp_path)
    old = path.read_text()
    save(tmp_path, implementation())
    old_manifest = path.parent / 'previous.json'
    old_manifest.write_text(old)
    assert 'source' not in load_checkpoint(old_manifest)
    assert 'source' in load_checkpoint(path)


def test_assigned_difficulty_cannot_be_downgraded_on_resume(tmp_path):
    from tests.generation.test_blueprint_pagination import fixture
    plan, source, req = fixture()
    req['difficulty_profile'] = 'large_scale_retrieval'
    path = save_checkpoint(tmp_path, request=req, blueprint=plan, source=source,
                           evidence={'status': 'failed', 'attempts': []})
    assert load_checkpoint(path, expected_request=req)['request']['difficulty_profile'] == 'large_scale_retrieval'
    with pytest.raises(ValueError, match='request differs'):
        load_checkpoint(path, expected_request={**req, 'difficulty_profile': 'standard'})


@pytest.mark.parametrize('tamper', ['content', 'path', 'symlink'])
def test_checkpoint_cannot_accept_modified_or_redirected_source(tmp_path, tamper):
    path = save(tmp_path, implementation())
    manifest = json.loads(path.read_text())
    target = path.parent / manifest['files']['source']['filename']
    if tamper == 'content':
        target.write_text('{}')
    elif tamper == 'path':
        manifest['files']['source']['filename'] = '../source.json'
        path.write_text(json.dumps(manifest))
    else:
        other = tmp_path / 'redirected.json'
        target.replace(other)
        target.symlink_to(other)
    with pytest.raises(ValueError, match='AUTHOR_CHECKPOINT:'):
        load_checkpoint(path)


def test_resume_keeps_design_and_budget_but_repeats_execution_and_review(tmp_path):
    from unittest.mock import Mock, patch
    from env_factory.generation.code_agent import generate
    from env_factory.generation.semantic_review import CHECKS
    original = tmp_path / 'original'
    original.mkdir()
    task_request = dict(request(), require_bound_authorization=True,
        require_interaction_contract=True, require_business_requirements=True,
        require_structured_interactions=True)
    checkpoint = save_checkpoint(original, request=task_request, blueprint=blueprint(),
        source=implementation(), validation_feedback={'error': 'wrong count', 'feedback': {'issues': []}},
        evidence={'status': 'failed', 'author_seconds': 600,
            'attempts': [{'phase': 'implementation'}],
            'repair_budget': {'used': {'structural': 1, 'semantic': 0}}})
    destination = tmp_path / 'resumed'
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            assert 'Continue the SAME fixed business design' in prompt
            assert 'Read parent_validation.json' in prompt
            assert json.loads((destination / 'authoring/parent_validation.json').read_text())['error'] == 'wrong count'
            assert json.loads((destination / 'authoring/source.json').read_text()) == implementation()
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0, communicate=finish)
    review = {'status': 'pass', 'checks': {k: {'passed': True, 'evidence': 'checked'} for k in CHECKS},
        'findings': [], 'provenance': {'completed_turns': 1, 'usage': {}, 'events_sha256': 'hash'}}
    with patch('env_factory.generation.code_agent.subprocess.Popen', side_effect=launch) as author, \
         patch('env_factory.generation.code_agent.compile_source', return_value={'generation_pipeline': {}}) as compiler, \
         patch('env_factory.generation.code_agent.verify_delivery'), \
         patch('env_factory.generation.code_agent.review_source', return_value=review) as reviewer:
        result = generate(request=task_request, artifact_dir=destination, script_count=1, resume_from=checkpoint)
    assert author.call_count == compiler.call_count == reviewer.call_count == 1
    assert compiler.call_args.kwargs['blueprint'] == blueprint()
    assert result['generation_pipeline']['repair_budget']['used']['structural'] == 1
    renewed = load_checkpoint(destination / 'authoring_checkpoint')
    assert renewed['manifest']['source_attempts'] == 2
    assert renewed['validation_feedback']['error'] == 'wrong count'
    assert renewed['manifest']['author_seconds'] >= 600
    assert load_checkpoint(checkpoint)['manifest']['previous_status'] == 'failed'
    with pytest.raises(ValueError, match='fresh output'):
        generate(request=task_request, artifact_dir=destination, script_count=1, resume_from=checkpoint)
