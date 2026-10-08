import json
from unittest.mock import Mock, patch

import pytest

from env_factory.generation.code_agent import generate
from env_factory.generation.pipeline_errors import PipelineGenerationError
from env_factory.generate_task import _generation_failure_class


@pytest.mark.parametrize('persistent,budget,expected_calls', [(False, 30, 2), (True, 30, 2), (True, 6, 1)])
def test_capacity_recovery_keeps_model_and_shared_budget_without_business_repair(tmp_path, persistent, budget, expected_calls):
    clock, budgets, models = [0.0], [], []
    def launch(command, **kwargs):
        index = len(budgets)
        unavailable = persistent or index == 0
        def finish(prompt, timeout):
            models.append(command[command.index('--model') + 1])
            budgets.append(timeout)
            clock[0] += 3
            if unavailable:
                kwargs['stdout'].write(json.dumps({'type': 'turn.failed', 'error': {
                    'message': 'Selected model is at capacity. Please try a different model.'}})+'\n')
            else:
                (tmp_path/'authoring/source.json').write_text('{"version":"1.0"}')
                kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=1 if unavailable else 0, communicate=finish)
    review = {'status': 'pass', 'findings': [], 'provenance': {
        'completed_turns': 1, 'usage': {}, 'events_sha256': 'a'*64}}
    with patch('env_factory.generation.code_agent.subprocess.Popen', side_effect=launch), \
         patch('env_factory.generation.code_agent.time.monotonic', side_effect=lambda: clock[0]), \
         patch('env_factory.generation.code_agent.time.sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0]+seconds)), \
         patch('env_factory.generation.code_agent.compile_source', return_value={'generation_pipeline': {}}) as compiler, \
         patch('env_factory.generation.code_agent.verify_delivery'), \
         patch('env_factory.generation.code_agent.review_source', return_value=review):
        if persistent:
            with pytest.raises(PipelineGenerationError, match='CODE_AGENT_CAPACITY') as error:
                generate(request={}, artifact_dir=tmp_path, timeout=budget)
            assert _generation_failure_class(error.value) == 'INFRA'
            compiler.assert_not_called()
        else:
            generate(request={}, artifact_dir=tmp_path, timeout=budget)
            compiler.assert_called_once()
    assert models == ['gpt-6.1-sol'] * expected_calls
    assert budgets == ([budget, budget-8] if expected_calls == 2 else [budget])
    evidence = json.loads((tmp_path/'code_agent_generation.json').read_text())
    assert evidence['repair_budget']['used'] == {'structural': 0, 'semantic': 0}
    assert evidence['attempts'][0]['failure_code'] == 'CODE_AGENT_CAPACITY'
    assert evidence['attempts'][0]['business_attempt_consumed'] is False
    assert evidence['attempts'][0]['events_sha256']
    assert evidence['author_seconds'] <= budget
