import importlib.util,json,sys,types
from pathlib import Path
from unittest.mock import Mock,patch
import pytest

PROJECT=Path(__file__).resolve().parents[2]
def candidate():
    spec=importlib.util.spec_from_file_location('env_factory.generation.budget_candidate',PROJECT/'src/env_factory/generation/code_agent.py')
    module=importlib.util.module_from_spec(spec);module.__file__=str(PROJECT/'src/env_factory/generation/code_agent.py');spec.loader.exec_module(module)
    clock=[0.];module.time=types.SimpleNamespace(monotonic=lambda:clock[0], time=lambda:1_700_000_000 + clock[0])
    return module,clock

def test_exhausted_author_budget_prevents_another_model_invocation(tmp_path):
    m,clock=candidate()
    def launch(command,**kwargs):
        def finish(prompt,timeout):
            assert timeout==10
            clock[0]+=10
            (tmp_path/'authoring/source.json').write_text('{"version":"1.0"}')
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0,communicate=Mock(side_effect=finish))
    with patch.object(m.subprocess,'Popen',side_effect=launch) as launch_mock,patch.object(m,'compile_source',side_effect=ValueError('repair required')),patch.object(m,'review_source') as review:
        with pytest.raises(m.PipelineGenerationError,match='budget exhausted'):
            m.generate(request={},artifact_dir=tmp_path,timeout=10)
    assert launch_mock.call_count==1
    review.assert_not_called()
    evidence=json.loads((tmp_path/'code_agent_generation.json').read_text())
    assert evidence['author_seconds']==10 and evidence['status']=='failed'

@pytest.mark.parametrize('needs_kill',[False,True])
def test_author_hang_is_terminated_and_time_recorded(tmp_path,needs_kill):
    m,clock=candidate()
    def finish(prompt,timeout):
        clock[0]+=timeout
        (tmp_path/'code_agent_attempts/attempt-1/events.jsonl').write_text(
            '{"type":"error","message":"Reconnecting... 2/5 (request timed out)"}\n')
        raise m.subprocess.TimeoutExpired('test-author',timeout)
    process=Mock(pid=123456,communicate=Mock(side_effect=finish))
    if needs_kill:process.wait.side_effect=[m.subprocess.TimeoutExpired('test-author',5),None]
    with patch.object(m.subprocess,'Popen',return_value=process),patch.object(m.os,'killpg') as kill,patch.object(m,'compile_source') as compile_source:
        with pytest.raises(m.PipelineGenerationError,match='deadline exceeded'):
            m.generate(request={},artifact_dir=tmp_path,timeout=10)
    assert kill.call_args_list[0].args==(123456,m.signal.SIGTERM)
    assert kill.call_count==(2 if needs_kill else 1)
    if needs_kill:assert kill.call_args_list[1].args==(123456,m.signal.SIGKILL)
    compile_source.assert_not_called()
    evidence=json.loads((tmp_path/'code_agent_generation.json').read_text())
    assert evidence['author_seconds']==10
    assert evidence['attempts'][0]['author_seconds']==10
    assert evidence['attempts'][0]['transport']['reconnect_notices']==1
    assert evidence['attempts'][0]['transport']['network_seconds'] is None
    assert evidence['status']=='failed'


def test_parent_validation_does_not_consume_author_or_review_budget(tmp_path):
    m,clock=candidate();budgets=[];compilations=[]
    def launch(command,**kwargs):
        def finish(prompt,timeout):
            budgets.append(timeout);clock[0]+=3
            (tmp_path/'authoring/source.json').write_text(json.dumps({'version':'1.0', 'revision':len(budgets)}))
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0,communicate=Mock(side_effect=finish))
    def compile_source(*args,**kwargs):
        clock[0]+=900;compilations.append(True)
        if len(compilations)==1:raise ValueError('repair required')
        return {'generation_pipeline':{}}
    review={'status':'pass','findings':[],'provenance':{'completed_turns':1,'usage':{},'events_sha256':'a'*64}}
    with patch.object(m.subprocess,'Popen',side_effect=launch),patch.object(m,'compile_source',side_effect=compile_source),patch.object(m,'verify_delivery'),patch.object(m,'review_source',return_value=review) as reviewer:
        m.generate(request={},artifact_dir=tmp_path,timeout=10)
    evidence=json.loads((tmp_path/'code_agent_generation.json').read_text())
    assert budgets==[10,7]
    assert reviewer.call_args.kwargs['timeout']==180
    assert evidence['author_seconds']==6
    assert evidence['seconds']==1806
    assert [item['phase_seconds']['compile_and_execution'] for item in evidence['attempts']] == [900, 900]
    assert 'independent_review' not in evidence['attempts'][0]['phase_seconds']
    assert evidence['attempts'][1]['phase_seconds']['independent_review'] == 0


def test_failed_review_time_is_retained_without_consuming_author_budget(tmp_path):
    m, clock = candidate()
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            clock[0] += 2
            (tmp_path / 'authoring/source.json').write_text('{"version":"1.0"}')
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0, communicate=finish)
    def review(**kwargs):
        clock[0] += 180
        raise m.SemanticReviewUnavailable('review timed out')
    with patch.object(m.subprocess, 'Popen', side_effect=launch), \
         patch.object(m, 'compile_source', return_value={'generation_pipeline': {}}), \
         patch.object(m, 'verify_delivery'), patch.object(m, 'review_source', side_effect=review):
        with pytest.raises(m.PipelineGenerationError, match='review timed out'):
            m.generate(request={}, artifact_dir=tmp_path, timeout=10)
    evidence = json.loads((tmp_path / 'code_agent_generation.json').read_text())
    assert evidence['author_seconds'] == 2
    assert evidence['seconds'] == 182
    assert evidence['attempts'][0]['phase_seconds']['independent_review'] == 180


def test_blueprint_and_implementation_share_author_budget(tmp_path):
    from tests.generation.test_business_blueprint import blueprint, request
    m, clock = candidate()
    budgets = []
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            budgets.append(timeout)
            clock[0] += 6 if len(budgets) == 1 else 4
            workspace = tmp_path / 'authoring'
            if len(budgets) == 1:
                (workspace / 'blueprint.json').write_text(json.dumps(blueprint()))
            else:
                (workspace / 'source.json').write_text('{"version":"1.0"}')
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0, communicate=finish)
    with patch.object(m.subprocess, 'Popen', side_effect=launch), \
         patch.object(m, 'compile_source', side_effect=ValueError('repair required')), \
         patch.object(m, 'review_source') as review:
        with pytest.raises(m.PipelineGenerationError, match='budget exhausted'):
            m.generate(request=request(), artifact_dir=tmp_path, script_count=1, timeout=10)
    assert budgets == [10, 4]
    review.assert_not_called()
    evidence = json.loads((tmp_path / 'code_agent_generation.json').read_text())
    assert evidence['author_seconds'] == 10
    assert [item['phase'] for item in evidence['attempts']] == ['business_blueprint', 'implementation']
    from env_factory.generation.authoring_checkpoint import load_checkpoint
    checkpoint = load_checkpoint(Path(evidence['checkpoint']))
    assert checkpoint['blueprint'] == blueprint()
    assert checkpoint['manifest']['author_seconds'] == 10
    assert checkpoint['manifest']['previous_status'] == 'failed'


def test_transport_starved_repair_defers_without_consuming_an_invocation(tmp_path):
    from tests.generation.test_business_blueprint import blueprint, request
    from env_factory.generation.authoring_checkpoint import load_checkpoint
    m, clock = candidate()
    budgets = []
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            budgets.append(timeout)
            clock[0] += 200 if len(budgets) == 1 else 310
            workspace = tmp_path / 'authoring'
            (workspace / ('blueprint.json' if len(budgets) == 1 else 'source.json')).write_text(
                json.dumps(blueprint() if len(budgets) == 1 else {'version': '1.0'}))
            kwargs['stdout'].write('{"type":"error","message":"Reconnecting... 2/5 (request timed out)"}\n{"type":"turn.completed"}\n')
        return Mock(returncode=0, communicate=finish)
    with patch.object(m.subprocess, 'Popen', side_effect=launch) as author, \
         patch.object(m, 'compile_source', side_effect=ValueError('repair this field')):
        with pytest.raises(m.PipelineGenerationError, match='insufficient remaining author budget'):
            m.generate(request=request(), artifact_dir=tmp_path, script_count=1, timeout=600)
    assert author.call_count == 2
    evidence = json.loads((tmp_path / 'code_agent_generation.json').read_text())
    assert evidence['deferred_author_invocation']['remaining_seconds'] == 90
    checkpoint = load_checkpoint(tmp_path / 'authoring_checkpoint')
    assert checkpoint['manifest']['source_attempts'] == 1
    assert checkpoint['validation_feedback']['error'] == 'repair this field'


def test_active_validation_phase_is_published_before_work_and_cleared_on_failure(tmp_path):
    m, clock = candidate()
    def launch(command, **kwargs):
        def finish(prompt, timeout):
            clock[0] += 2
            (tmp_path / 'authoring/source.json').write_text('{"version":"1.0"}')
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(returncode=0, communicate=finish)
    def compile_source(*args, **kwargs):
        current = json.loads((tmp_path / 'code_agent_generation.json').read_text())
        assert current['attempts'][-1]['active_phase'] == 'compile_and_execution'
        assert current['attempts'][-1]['author_seconds'] == 2
        clock[0] += 30
        raise m.SemanticCalibrationUnavailable('judge unavailable')
    with patch.object(m.subprocess, 'Popen', side_effect=launch), \
         patch.object(m, 'compile_source', side_effect=compile_source):
        with pytest.raises(m.PipelineGenerationError, match='judge unavailable'):
            m.generate(request={}, artifact_dir=tmp_path, timeout=10)
    result = json.loads((tmp_path / 'code_agent_generation.json').read_text())
    assert 'active_phase' not in result['attempts'][-1]
    assert result['attempts'][-1]['phase_seconds']['compile_and_execution'] == 30
