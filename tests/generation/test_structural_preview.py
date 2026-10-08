import json
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source, main
from .test_agent_authoring import fixture


def test_preview_never_calls_judge_or_marks_task_ready(tmp_path):
    source, request = fixture("multi_step_agentic")
    with patch("env_factory.generation.agent_authoring.verify_execution", side_effect=AssertionError("must not execute")):
        result = compile_source(source, request=request, root=tmp_path, structural_preview=True)
    assert result["task_readiness"]["ready"] is False
    report = result["generation_pipeline"]["verification"]
    assert report["structural_checks_passed"] is True
    assert report["passed"] is False
    assert "semantic_calibration" in report["pending"]


def test_default_compilation_still_requires_execution(tmp_path):
    source, request = fixture("multi_step_agentic")
    with patch("env_factory.generation.agent_authoring.verify_execution", side_effect=ValueError("execution failed")):
        with pytest.raises(ValueError, match="execution failed"):
            compile_source(source, request=request, root=tmp_path)


def test_preview_cli_writes_only_uncertified_report(tmp_path):
    source, request = fixture("multi_step_agentic")
    source_path, request_path = tmp_path / "source.json", tmp_path / "request.json"
    source_path.write_text(json.dumps(source))
    request_path.write_text(json.dumps(request))
    output = tmp_path / "preview"
    argv = ["authoring", "--source", str(source_path), "--request", str(request_path),
            "--output", str(output), "--structural-preview"]
    with patch("sys.argv", argv):
        main()
    assert json.loads((output / "structural_preview.json").read_text())["passed"] is False
    assert not (output / "task.json").exists()
    assert not (output / "compiled_artifacts.json").exists()
    (output / "task.json").write_text("{}")
    with patch("sys.argv", argv), pytest.raises(SystemExit):
        main()


def test_failed_cli_rerun_archives_previous_delivery(tmp_path):
    source, request = fixture("multi_step_agentic")
    source_path, request_path = tmp_path / "source.json", tmp_path / "request.json"
    source_path.write_text(json.dumps(source))
    request_path.write_text(json.dumps(request))
    output = tmp_path / "delivery"
    output.mkdir()
    names = ("task.json", "compiled_artifacts.json", "status.json", "buildability.json")
    for name in names:
        (output / name).write_text('{"previous": true}')
    argv = ["authoring", "--source", str(source_path), "--request", str(request_path),
            "--output", str(output)]
    with patch("sys.argv", argv), patch(
            "env_factory.generation.agent_authoring.verify_execution",
            side_effect=ValueError("new reference failed")):
        with pytest.raises(ValueError, match="new reference failed"):
            main()
    for name in names:
        assert not (output / name).exists()
        archived = list((output / ".previous_deliveries").glob("*/" + name))
        assert len(archived) == 1
        assert json.loads(archived[0].read_text()) == {"previous": True}


@pytest.mark.parametrize('outcome', ['reject', 'interrupt', 'success'])
def test_buildability_failure_does_not_publish_rejected_task(tmp_path, outcome):
    from types import SimpleNamespace
    source, request = fixture('multi_step_agentic')
    source_path, request_path = tmp_path / 'source.json', tmp_path / 'request.json'
    source_path.write_text(json.dumps(source))
    request_path.write_text(json.dumps(request))
    output = tmp_path / 'delivery'
    argv = ['authoring', '--source', str(source_path), '--request', str(request_path),
            '--output', str(output)]

    def assess(*args, **kwargs):
        assert (output / 'task.json').is_file()
        (output / 'buildability.json').write_text(json.dumps({'passed': outcome == 'success'}))
        if outcome == 'interrupt':
            raise KeyboardInterrupt()
        return SimpleNamespace(returncode=0 if outcome == 'success' else 2)

    with patch('sys.argv', argv), patch('env_factory.generation.delivery_preflight.verify_delivery'), \
            patch('env_factory.generation.agent_authoring.subprocess.run', side_effect=assess):
        if outcome == 'success':
            main()
        else:
            with pytest.raises(KeyboardInterrupt if outcome == 'interrupt' else SystemExit):
                main()
    for name in ('task.json', 'compiled_artifacts.json', 'buildability.json'):
        if outcome == 'success':
            assert (output / name).is_file()
        else:
            assert not (output / name).exists()
            archived = list((output / '.failed_deliveries').glob('*/' + name))
            assert len(archived) == 1
            assert json.loads(archived[0].read_text())


def test_lifecycle_unknown_tool_rejected_before_execution(tmp_path):
    from env_factory.sandbox_runtime import SandboxError
    source,request=fixture('multi_step_agentic')
    source['business_lifecycle']={'version':'1.0','tools':{'missing_tool':{'roles':['buyer']}}}
    with pytest.raises(SandboxError,match='unknown lifecycle tool'):
        compile_source(source,request=request,root=tmp_path,structural_preview=True)


@pytest.mark.parametrize('operand_location', ['selector', 'condition'])
def test_lifecycle_guard_unknown_argument_rejected_in_structural_preview(tmp_path, operand_location):
    source, request = fixture('multi_step_agentic')
    guard = {'id': 'stock', 'table': 'zones', 'selector': {'id': 821},
             'conditions': {'temperature': {'gte': 0}}}
    if operand_location == 'selector':
        guard['selector']['id'] = {'argument': 'undefined_argument'}
    else:
        guard['conditions']['temperature']['gte'] = {'argument': 'undefined_argument'}
    source['business_lifecycle'] = {'version': '1.0', 'tools': {'read_temperature': {'guards': [guard]}}}
    with pytest.raises(ValueError, match='guards must bind required tool arguments'):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)


def test_read_checkpoint_compiles_and_runs_real_reference_and_ablations(tmp_path):
    source, request = fixture('multi_step_agentic')
    target = source['metric_implementations'][0]['expected']['targets'][0]
    target['expression'] = {'at_tool': {'name': 'read_temperature',
                                      'expression': target['expression']}}
    result = compile_source(source, request=request, root=tmp_path)
    assert result['generation_pipeline']['verification']['passed']
    checks = result['generation_pipeline']['verification']['checks']
    assert checks['reference_0'] and checks['no_tools_0']
    assert checks['skip_0_1']


def test_preview_rejects_checkpoint_from_undeclared_tool(tmp_path):
    from env_factory.sandbox_runtime import SandboxError
    source, request = fixture('multi_step_agentic')
    target = source['metric_implementations'][0]['expected']['targets'][0]
    target['expression'] = {'at_tool': {'name': 'missing', 'expression': target['expression']}}
    with pytest.raises(SandboxError, match='read-only'):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)


@pytest.mark.parametrize('preview', [True, False])
def test_ungrounded_reward_rejected_before_any_model_execution(tmp_path, preview):
    source, request = fixture('simple_agentic')
    from copy import deepcopy
    source['metrics'][0]['weight'] = 0.5
    evidence = deepcopy(source['metrics'][0])
    evidence['id'] = 'private_evidence'
    source['metrics'].append(evidence)
    rule = deepcopy(source['metric_implementations'][0])
    rule['metric_id'] = 'private_evidence'
    source['metric_implementations'].append(rule)
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {'literal': 18}
    source['answer_contract'] = {'format': 'json_object', 'schema': {'type': 'object',
        'properties': {'temperature_answer': {'type': 'number'}, 'private_evidence': {'type': 'number'}},
        'required': ['temperature_answer', 'private_evidence'], 'additionalProperties': False}}
    for rule in source['metric_implementations']:
        target = rule['expected']['targets'][0]
        rule.update(operator='value_targets', expected={'answer_format': 'json_object',
            'targets': [{'key': rule['metric_id'], 'expression': target['expression']}]})
    source['scenarios'][0]['steps'][-1]['content'] = '{"temperature_answer":18,"private_evidence":18}'
    with patch('env_factory.generation.agent_authoring.verify_execution',
               side_effect=AssertionError('must reject before model execution')):
        with pytest.raises(ValueError, match='REWARD_TARGET_QUALIFICATION.*temperature_answer.*no private data dependency'):
            compile_source(source, root=tmp_path, request=request, structural_preview=preview)


def test_literal_direct_answer_remains_supported_in_preview(tmp_path):
    source, request = fixture('direct_response')
    result = compile_source(source, root=tmp_path, request=request, structural_preview=True)
    assert result['generation_pipeline']['verification']['structural_checks_passed']
