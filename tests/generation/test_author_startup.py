import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from env_factory.generation import author_startup as startup
from .test_author_budget import candidate
from .test_business_blueprint import blueprint, implementation, request


@pytest.mark.parametrize('activity', ['cli', 'artifact'])
def test_active_initial_author_keeps_same_process_and_original_deadline(tmp_path, monkeypatch, activity):
    clock = [0.]
    monkeypatch.setattr(startup, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]))
    events = tmp_path / 'events.jsonl'
    events.write_text('')
    before = startup.workspace_signature(tmp_path)
    calls = []
    def communicate(prompt=None, *, timeout):
        calls.append((prompt, timeout))
        if len(calls) == 1:
            clock[0] = 120
            if activity == 'cli':
                events.write_text(json.dumps({'type': 'item.started', 'item': {'type': 'command_execution'}}))
            else:
                (tmp_path / 'blueprint.json').write_text('{}')
            raise subprocess.TimeoutExpired('author', timeout)
        assert prompt is None  # Never send the input twice.
        clock[0] += 200
    attempt = {}
    startup.communicate(SimpleNamespace(communicate=communicate), 'same prompt', budget=600,
        events=events, stdout=Mock(), workspace=tmp_path, before=before, attempt=attempt)
    assert calls == [('same prompt', 120), (None, 480)]
    assert attempt['startup_check']['elapsed_seconds'] == 120


def test_workspace_probe_does_not_follow_symlinks(tmp_path):
    (tmp_path / 'source.json').symlink_to(tmp_path / 'absent')
    assert startup.workspace_signature(tmp_path) == {'source.json': 'symlink'}


@pytest.mark.parametrize('both_stall', [False, True])
def test_one_silent_startup_recovery_keeps_request_budget_and_business_attempt_limits(tmp_path, both_stall):
    m, clock = candidate()
    invocations = []
    def launch(command, **kwargs):
        index = len(invocations) + 1
        record = {'index': index}
        invocations.append(record)
        def finish(prompt, timeout):
            record.update(prompt=prompt, timeout=timeout)
            if index == 1 or both_stall:
                kwargs['stdout'].write('{"type":"thread.started"}\n{"type":"turn.started"}\n')
                clock[0] += timeout
                raise subprocess.TimeoutExpired('author', timeout)
            workspace = tmp_path / 'authoring'
            (workspace / ('blueprint.json' if index == 2 else 'source.json')).write_text(
                json.dumps(blueprint() if index == 2 else implementation()))
            clock[0] += 10
            kwargs['stdout'].write('{"type":"turn.completed"}\n')
        return Mock(pid=123456, returncode=0, communicate=finish)
    artifact = {'generation_pipeline': {'verification': {}}, 'requirement_coverage': {'status': 'not_declared'}}
    from env_factory.generation.semantic_review import CHECKS
    review = {'status': 'pass', 'checks': {k: {'passed': True, 'evidence': 'checked'} for k in CHECKS},
        'findings': [], 'provenance': {'completed_turns': 1, 'usage': {}, 'events_sha256': 'hash'}}
    with patch.object(startup, 'time', m.time), patch.object(m.subprocess, 'Popen', side_effect=launch), \
         patch.object(m.os, 'killpg') as kill, patch.object(m, 'compile_source', return_value=artifact) as compile_task, \
         patch.object(m, 'verify_delivery', return_value={'passed': True}), \
         patch.object(m, 'review_source', return_value=review):
        if both_stall:
            with pytest.raises(m.PipelineGenerationError, match='CODE_AGENT_STARTUP_STALLED'):
                m.generate(request=request(), artifact_dir=tmp_path, script_count=1, timeout=600)
            compile_task.assert_not_called()
        else:
            m.generate(request=request(), artifact_dir=tmp_path, script_count=1, timeout=600)
            assert compile_task.call_count == 1
    evidence = json.loads((tmp_path / 'code_agent_generation.json').read_text())
    assert evidence['startup_recoveries'] == 1
    assert len(invocations) == (2 if both_stall else 3)
    assert evidence['author_seconds'] == (240 if both_stall else 140)
    assert invocations[1]['prompt'].endswith(invocations[0]['prompt'])
    assert evidence['attempts'][1]['budget_seconds'] == 480
    assert kill.call_count == (2 if both_stall else 1)
    assert evidence['attempts'][0]['events_sha256']
    assert evidence['attempts'][0]['startup_check']['substantive_cli_activity'] is False
    assert evidence['attempts'][0]['startup_check']['workspace_changed'] is False
    assert all(value == 0 for value in evidence['repair_budget']['used'].values())
    assert not any(a.get('actionable_feedback_at') for a in evidence['attempts'])
    if not both_stall:
        assert evidence['status'] == 'passed'
        assert evidence['attempts'][-1]['budget_seconds'] == 470
    else:
        from env_factory.generate_task import _generation_failure_class
        assert _generation_failure_class(m.PipelineGenerationError(evidence['error'])) == 'GEN_STARTUP_STALL'
