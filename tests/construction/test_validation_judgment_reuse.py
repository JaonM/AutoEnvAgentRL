import json
import sys
import types
from unittest.mock import patch

import pytest

from env_factory import sandbox_runtime
from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.semantic_reward import compile_semantic_outcome
from scripts.sandbox import generate_sandbox_scaffold as scaffold
from scripts.sandbox import validate_agentic_training_value as validator
from tests.generation.test_agent_authoring import fixture
from tests.generation.test_semantic_reward import fixture as semantic_fixture


def generated_module(tmp_path, monkeypatch):
    source, request = fixture('direct_response')
    artifact = compile_source(source, root=tmp_path, request=request, structural_preview=True)
    item, schema = semantic_fixture()
    metric, _ = compile_semantic_outcome(item, schema=schema, weight=1)
    artifact['metrics'] = [metric]
    artifact['metric_implementations'] = []
    (tmp_path / 'BUILD_CONTRACT.json').write_text(json.dumps({**artifact, 'artifacts': artifact}))
    scaffold.generate(tmp_path)
    monkeypatch.setitem(sys.modules, 'sandbox_runtime', sandbox_runtime)
    hooks = types.ModuleType('task_impl')
    exec(scaffold.TASK_IMPL_SOURCE, hooks.__dict__)
    monkeypatch.setitem(sys.modules, 'task_impl', hooks)
    app = types.ModuleType('generated_app')
    app.__file__ = str(tmp_path / 'app.py')
    exec(scaffold.APP_SOURCE, app.__dict__)
    return app


@pytest.mark.parametrize('validation,expected_calls', [(False, 2), (True, 1)])
def test_generated_live_episodes_are_isolated_but_local_validation_can_reuse(tmp_path, monkeypatch, validation, expected_calls):
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK', '0')
    monkeypatch.setenv('SANDBOX_LLM_API_KEY', 'offline-test-key')
    monkeypatch.setenv('SANDBOX_LLM_MODEL', 'offline-test-model')
    module = generated_module(tmp_path, monkeypatch)
    app = (validator.import_app(tmp_path, reuse_validation_judgments=True)
           if validation else module.create_app())
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as judge:
        for i in range(2):
            app.episode_store.reset(episode_id=str(i), seed=17)
            app.episode_store.set_state('final_agent_response', '{"why":"The public facts explain the answer."}')
            app.reward_callback()
        assert judge.call_count == expected_calls
        # Changed answer must be judged anew even within this validation run.
        app.episode_store.reset(episode_id='changed', seed=17)
        app.episode_store.set_state('final_agent_response', '{"why":"A different answer."}')
        app.reward_callback()
        assert judge.call_count == expected_calls + 1
    report = app.validation_judgment_report()
    assert report['enabled'] is validation
    assert report['reused_judgments'] == int(validation)
    assert 'offline-test-key' not in str(report)


def test_validator_supports_existing_app_factory_without_reuse_parameter(tmp_path):
    (tmp_path / 'app.py').write_text('def create_app():\n    return "existing-app"\n')
    assert validator.import_app(tmp_path, reuse_validation_judgments=True) == 'existing-app'
