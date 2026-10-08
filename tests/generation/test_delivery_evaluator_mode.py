import json
from pathlib import Path
from unittest.mock import patch, Mock
import subprocess
import pytest
from env_factory.generation.agent_authoring import compile_source
from tests.generation.test_agent_authoring import fixture

from env_factory.generation import delivery_preflight as m

@pytest.mark.parametrize('semantic', [False, True])
def test_delivery_selects_real_judge_for_semantics(tmp_path, semantic):
    source, request = fixture('simple_agentic')
    artifacts = compile_source(source, root=tmp_path, request=request)
    if semantic:
        artifacts['metrics'][0].update(type='model-based', evaluator={'kind':'external_llm_judge', 'source':'external_llm', 'score_mapping':{'pass':1,'fail':0}})
    def run(command, **kwargs):
        assert command[-2:] == ['--evaluator-mode', 'live' if semantic else 'mock']
        assert kwargs['env']['SANDBOX_EVALUATOR_MOCK'] == ('0' if semantic else '1')
        assert kwargs['timeout'] == (900 if semantic else 90)
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'curriculum_training_ready': True, 'failures': []}))
        return subprocess.CompletedProcess(command, 0)
    with patch.object(m.subprocess, 'run', side_effect=run), \
         patch.object(m, 'score_file', return_value=Mock(eligible=True)):
        assert m.verify_delivery(artifacts, tmp_path)['evaluator_mode'] == ('live' if semantic else 'mock')

@pytest.mark.parametrize('failure', ['timeout', 'fallback'])
def test_semantic_infrastructure_does_not_become_author_repair(tmp_path, failure):
    source, request = fixture('simple_agentic')
    artifacts = compile_source(source, root=tmp_path, request=request)
    artifacts['metrics'][0].update(type='model-based', evaluator={'kind':'external_llm_judge', 'source':'external_llm', 'score_mapping':{'pass':1,'fail':0}})
    def run(command, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        Path(command[command.index('--output') + 1]).write_text(json.dumps({
            'curriculum_training_ready': False, 'failures': [{'gate': 'evaluator_availability'}]}))
        return subprocess.CompletedProcess(command, 1)
    with patch.object(m.subprocess, 'run', side_effect=run), \
         patch.object(m, 'score_file', return_value=Mock(eligible=True)):
        with pytest.raises(m.SemanticCalibrationUnavailable):
            m.verify_delivery(artifacts, tmp_path)


def test_delivery_checks_actual_build_eligibility_before_subprocess(tmp_path):
    source, request = fixture('simple_agentic')
    artifacts = compile_source(source, root=tmp_path, request=request)
    artifacts['metric_implementations'][0]['expected']['targets'][0]['expression'] = {'literal': 18}
    with patch.object(m.subprocess, 'run') as run:
        with pytest.raises(ValueError, match='AGENT_TASK_QUALIFICATION_FAILED'):
            m.verify_delivery(artifacts, tmp_path)
    run.assert_not_called()
