import json
import pytest
from unittest.mock import patch

from env_factory_v2.author_mcp import AuthorMCP
from env_factory_v2.author_session import author_command, checked_source
from env_factory_v2.contract import ContractError
from env_factory_v2.examples import direct
from env_factory_v2.code_agent import NativeCodeAgent


def test_author_command_limits_tools_and_preserves_requested_model(tmp_path):
    args = author_command(tmp_path)
    assert args[args.index('--model')+1] == 'gpt-6.1-sol'
    assert args[args.index('--sandbox')+1] == 'read-only'
    assert '--ignore-user-config' in args
    assert 'shell_tool' in args and 'unified_exec' in args and 'multi_agent' in args
    assert 'mcp_servers.envfactory_author.required=true' in args
    assert 'mcp_servers.envfactory_author.default_tools_approval_mode="approve"' in args


def test_host_rejects_old_check_after_draft_change(tmp_path):
    server = AuthorMCP(tmp_path/'workbench', source=direct())
    def call(name, arguments):
        server.dispatch({'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}})
    call('check_contract', {})
    assert checked_source(tmp_path)[0] == direct()
    call('put_section', {'section': 'task', 'value_json': json.dumps('new task')})
    with pytest.raises(ContractError, match='current draft'):
        checked_source(tmp_path)


def test_native_generator_selects_host_tools_without_changing_judge_transport(tmp_path):
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a'*24,
                            timeout=255, evidence_root=tmp_path)
    provenance = {'source_identity': 'test', 'model': 'gpt-6.1-sol'}
    with patch('env_factory_v2.author_session.generate_in_session', return_value=(direct(), provenance)) as run:
        result, _ = agent.generate({'goal': direct()['task']}, author_tools=True,
                                   feedback={'previous_source': direct(), 'error': 'actual failure'})
    assert result == direct()
    assert run.call_args.kwargs['previous_source'] == direct()
    assert run.call_args.kwargs['feedback'] == {'error': 'actual failure'}
    assert run.call_args.kwargs['timeout'] == 240
    assert len(agent.provenance) == 1 and agent.base_url == 'http://localhost:9999/v1'


def test_container_cannot_start_a_host_author_session(tmp_path):
    agent = NativeCodeAgent(base_url='http://host.docker.internal:9999/v1', token='a'*24,
                            evidence_root=tmp_path)
    with pytest.raises(ContractError, match='only on the host'):
        agent.generate({'goal': 'anything'}, author_tools=True)
