import pytest

from env_factory.generation import codex_transport as module
transport_args = module.transport_args


def test_auto_preserves_existing_provider_without_reading_config(tmp_path):
    (tmp_path / 'config.toml').write_text('not valid toml')
    assert transport_args({'CODEX_HOME':str(tmp_path)}) == []


def test_https_preserves_authentication_and_never_adds_credentials(tmp_path):
    args=transport_args({'CODEX_HOME':str(tmp_path),'ENVFACTORY_CODEX_TRANSPORT':'https'})
    assert 'model_providers.envfactory_openai_https.requires_openai_auth=true' in args
    assert 'model_providers.envfactory_openai_https.supports_websockets=false' in args
    assert all('base_url' not in a and 'env_key' not in a and 'token' not in a for a in args)


@pytest.mark.parametrize('config', ['model_provider="custom"', 'profile="work"',
    'openai_base_url="https://example.org/v1"', 'chatgpt_base_url="https://example.org"'])
def test_https_does_not_replace_custom_routing(tmp_path, config):
    (tmp_path / 'config.toml').write_text(config)
    with pytest.raises(ValueError,match='preserve customized'):
        transport_args({'CODEX_HOME':str(tmp_path),'ENVFACTORY_CODEX_TRANSPORT':'https'})


def test_invalid_transport_fails_closed():
    with pytest.raises(ValueError,match='auto or https'):
        transport_args({'ENVFACTORY_CODEX_TRANSPORT':'htps'})


def test_custom_endpoint_environment_is_preserved(tmp_path):
    with pytest.raises(ValueError,match='preserve customized'):
        transport_args({'CODEX_HOME':str(tmp_path),'ENVFACTORY_CODEX_TRANSPORT':'https',
                        'OPENAI_BASE_URL':'https://example.org/v1'})


def test_author_and_semantic_review_preserve_model_sandbox_and_schema(tmp_path, monkeypatch):
    from env_factory.generation.agent_cli import command
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))
    monkeypatch.setenv('ENVFACTORY_CODEX_TRANSPORT', 'https')
    for readonly in (False, True):
        args = command('codex', 'gpt-6-astra', 'input', tmp_path / 'response',
                       readonly=readonly, schema=tmp_path / 'schema')
        assert 'model_providers.envfactory_openai_https.supports_websockets=false' in args
        assert args[args.index('--model') + 1] == 'gpt-6-astra'
        assert args[args.index('--sandbox') + 1] == ('read-only' if readonly else 'workspace-write')
        assert args[args.index('--output-schema') + 1] == str(tmp_path / 'schema')
        assert args[-1] == '-'


def test_transport_setting_does_not_change_other_agents(tmp_path, monkeypatch):
    from env_factory.generation.agent_cli import command
    monkeypatch.setenv('ENVFACTORY_CODEX_TRANSPORT', 'invalid')
    for agent in ('claude', 'opencode'):
        assert command(agent, 'model', 'prompt', tmp_path / 'response')[0] == agent


def test_shell_entry_execs_codex_with_original_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))
    monkeypatch.setenv('ENVFACTORY_CODEX_TRANSPORT', 'https')
    monkeypatch.setattr(module.sys, 'argv', ['wrapper', 'exec', '--approve-for-me', '--model', 'gpt-6-astra', 'prompt with spaces'])
    calls=[]
    monkeypatch.setattr(module.os, 'execvp', lambda exe, args: calls.append((exe, args)))
    module.main()
    assert calls[0][0] == 'codex'
    assert calls[0][1][-5:] == ['exec', '--approve-for-me', '--model', 'gpt-6-astra', 'prompt with spaces']
