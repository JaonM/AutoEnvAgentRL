import json
import pytest
from env_factory.generation.agent_cli import command, completed_events

@pytest.mark.parametrize('agent', ['codex', 'claude', 'opencode'])
def test_command_selects_cli_model(agent, tmp_path):
    args = command(agent, 'provider/model', 'hello', tmp_path / 'response')
    assert args[0] == agent
    assert args[args.index('--model') + 1] == 'provider/model'

@pytest.mark.parametrize('agent,event', [
    ('claude', {'type': 'result', 'subtype': 'success', 'result': '{}', 'usage': {'input_tokens': 12}}),
    ('opencode', {'type': 'step_finish', 'part': {'reason': 'stop', 'tokens': {'input': 12}}}),
])
def test_completion_normalization(agent, event, tmp_path):
    response = tmp_path / 'response'
    text = json.dumps(event)
    if agent == 'opencode':
        text = json.dumps({'type': 'text', 'part': {'text': '{}'}}) + '\n' + text
    result = completed_events(agent, text, response)
    assert len(result) == 1
    assert response.read_text() == '{}'

@pytest.mark.parametrize('agent,event', [
    ('claude', {'type': 'result', 'subtype': 'error_max_turns', 'is_error': True}),
    ('opencode', {'type': 'step_finish', 'part': {'reason': 'tool-calls'}}),
    ('opencode', {'type': 'error', 'error': 'disconnected'}),
])
def test_incomplete_or_failed_turn_is_not_success(agent, event, tmp_path):
    assert completed_events(agent, json.dumps(event), tmp_path / 'response') == []
