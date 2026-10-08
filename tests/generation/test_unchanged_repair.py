import json
from unittest.mock import Mock, patch

import pytest

from tests.generation.test_author_budget import candidate


@pytest.mark.parametrize('second', ['{"version":"1.0","value":1}', '{\n "value": 1, "version": "1.0"\n}'])
def test_unchanged_repair_stops_before_repeated_execution(tmp_path, second):
    module, clock = candidate()
    sources = iter(['{"version":"1.0","value":1}', second])
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            clock[0] += 1
            (tmp_path / 'authoring/source.json').write_text(next(sources))
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0, communicate=finish)
    with patch.object(module.subprocess, 'Popen', side_effect=launch) as author, \
         patch.object(module, 'compile_source', side_effect=ValueError('missing conditional route')) as compiler, \
         patch.object(module, 'review_source') as review:
        with pytest.raises(module.PipelineGenerationError, match='CODE_AGENT_REPAIR_NO_CHANGE'):
            module.generate(request={}, artifact_dir=tmp_path, timeout=30)
    assert author.call_count == 2
    assert compiler.call_count == 1
    review.assert_not_called()
    evidence = json.loads((tmp_path / 'code_agent_generation.json').read_text())
    assert evidence['attempts'][-1]['repair_stop_reason'] == 'unchanged_rejected_source'
    assert evidence['attempts'][-1]['issue_ids'] == evidence['attempts'][0]['issue_ids']
    assert 'compile_and_execution' not in evidence['attempts'][-1].get('phase_seconds', {})


def test_changed_repair_still_requires_fresh_full_validation(tmp_path):
    module, clock = candidate()
    counter = []
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            counter.append(1)
            clock[0] += 1
            (tmp_path / 'authoring/source.json').write_text(json.dumps({'version': '1.0', 'value': len(counter)}))
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0, communicate=finish)
    review = {'status': 'pass', 'findings': [], 'provenance': {'completed_turns': 1,
              'usage': {}, 'events_sha256': 'a' * 64}}
    with patch.object(module.subprocess, 'Popen', side_effect=launch), \
         patch.object(module, 'compile_source', side_effect=[ValueError('missing route'), {'generation_pipeline': {}}]) as compiler, \
         patch.object(module, 'verify_delivery') as delivery, \
         patch.object(module, 'review_source', return_value=review) as reviewer:
        module.generate(request={}, artifact_dir=tmp_path, timeout=30)
    assert compiler.call_count == 2
    delivery.assert_called_once()
    reviewer.assert_called_once()


def test_rejected_source_digest_survives_checkpoint_resume(tmp_path):
    from env_factory.generation.authoring_checkpoint import save_checkpoint, load_checkpoint
    from env_factory.generation.code_agent import generate, _canonical_source_digest
    from tests.generation.test_business_blueprint import blueprint, request, implementation
    original = tmp_path / 'original'
    original.mkdir()
    task_request = dict(request(), require_bound_authorization=True,
        require_interaction_contract=True, require_business_requirements=True,
        require_structured_interactions=True)
    digest = _canonical_source_digest(implementation())
    checkpoint = save_checkpoint(original, request=task_request, blueprint=blueprint(),
        source=implementation(), validation_feedback={'error': 'wrong count',
            'rejected_source_canonical_sha256': digest},
        evidence={'status': 'failed', 'attempts': [{'phase': 'implementation', 'issue_ids': ['count']}],
            'repair_budget': {'used': {'structural': 1, 'semantic': 0}}})
    destination = tmp_path / 'resumed'
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            # The restored source is deliberately returned unchanged.
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0, communicate=finish)
    with patch('env_factory.generation.code_agent.subprocess.Popen', side_effect=launch), \
         patch('env_factory.generation.code_agent.compile_source') as compiler:
        with pytest.raises(ValueError, match='CODE_AGENT_REPAIR_NO_CHANGE'):
            generate(request=task_request, artifact_dir=destination, script_count=1, resume_from=checkpoint)
    compiler.assert_not_called()
    saved = load_checkpoint(destination / 'authoring_checkpoint')
    assert saved['validation_feedback']['rejected_source_canonical_sha256'] == digest
    assert saved['manifest']['source_attempts'] == 2
