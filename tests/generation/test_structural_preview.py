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
