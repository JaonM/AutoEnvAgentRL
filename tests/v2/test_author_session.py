import json
import pytest
from unittest.mock import patch

from env_factory_v2.authoring.author_mcp import AuthorMCP
from env_factory_v2.authoring.author_session import author_command, checked_source
from env_factory_v2.contracts.contract import ContractError
from env_factory_v2.examples import direct
from env_factory_v2.native.code_agent import NativeCodeAgent


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
    with patch('env_factory_v2.authoring.author_session.generate_in_session', return_value=(direct(), provenance)) as run:
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


def test_incomplete_author_preserves_current_draft_and_failure_check(tmp_path):
    from env_factory_v2.authoring.author_session import incomplete_draft
    draft = {'environment': {'version': '1.0', 'id': 'partial'}}
    server = AuthorMCP(tmp_path / 'workbench', source=draft)
    server.dispatch({'id': 1, 'method': 'tools/call',
                     'params': {'name': 'check_contract', 'arguments': {}}})
    error = incomplete_draft(tmp_path, 'not complete')
    assert error.recovery['previous_source'] == draft
    assert not error.recovery['latest_operation']['result']['passed']
    assert json.loads((tmp_path / 'failure.json').read_text()) == error.recovery


def test_stale_success_receipt_is_not_attached_to_new_draft(tmp_path):
    from env_factory_v2.authoring.author_session import incomplete_draft
    server = AuthorMCP(tmp_path / 'workbench', source=direct())
    server.dispatch({'id': 1, 'method': 'tools/call',
                     'params': {'name': 'check_contract', 'arguments': {}}})
    changed = direct()
    changed['task'] = 'Changed task without check'
    (tmp_path / 'workbench/draft.json').write_text(json.dumps(changed))
    recovery = incomplete_draft(tmp_path, 'stale check').recovery
    assert recovery['previous_source'] == changed
    assert 'latest_operation' not in recovery
    assert recovery['scope'] == 'unqualified_author_draft'


def test_invalid_draft_is_not_used_as_repair_source(tmp_path):
    from env_factory_v2.authoring.author_session import incomplete_draft
    (tmp_path / 'workbench').mkdir()
    (tmp_path / 'workbench/draft.json').write_text('{"unexpected":true}')
    assert 'previous_source' not in incomplete_draft(tmp_path, 'failed').recovery


def test_environment_only_draft_is_forwarded_to_next_author_session(tmp_path):
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a'*24, evidence_root=tmp_path)
    partial = {'environment': {'version': '1.0', 'id': 'draft'}}
    with patch('env_factory_v2.authoring.author_session.generate_in_session',
               return_value=(direct(), {'source_identity': 'test'})) as generate:
        agent.generate({'goal': 'test'}, author_tools=True, feedback={'previous_source': partial})
    assert generate.call_args.kwargs['previous_source'] == partial


def test_pipeline_hands_off_partial_work_within_existing_budget(tmp_path):
    from unittest.mock import Mock
    from env_factory_v2.authoring.author_session import AuthorDraftError
    from env_factory_v2.pipeline import build

    partial = {'environment': {'version': '1.0', 'id': 'draft'}}
    agent = Mock()
    agent.generate.side_effect = [AuthorDraftError('not complete', {
        'previous_source': partial, 'scope': 'unqualified_author_draft', 'error': 'missing task'}),
        RuntimeError('capacity unavailable')]
    output = tmp_path / 'build'
    result = build({'goal': 'test'}, agent=agent, output=output, repairs=2, author_tools=True)
    assert agent.generate.call_count == 2  # infrastructure failure does not start a third repair
    feedback = agent.generate.call_args.kwargs['feedback']
    assert feedback['previous_source'] == partial
    assert feedback['author_recovery']['error'] == 'missing task'
    assert result['status'] == 'infrastructure_failed'
    assert not result['qualified']
    assert json.loads((output / 'attempt-1/author_draft.json').read_text()) == partial


def test_completed_but_unfinished_session_raises_recoverable_draft(tmp_path):
    from pathlib import Path
    from unittest.mock import Mock
    from env_factory_v2.authoring.author_session import AuthorDraftError, generate_in_session

    draft = {'version': '2.0', 'id': 'unfinished'}
    def process(args, **kwargs):
        folder = Path(args[args.index('--output-last-message') + 1]).parent
        server = AuthorMCP(folder / 'workbench', source=draft)
        server.dispatch({'id': 1, 'method': 'tools/call',
                         'params': {'name': 'check_contract', 'arguments': {}}})
        (folder / 'events.jsonl').write_text(json.dumps({'type': 'turn.completed'}) + '\n')
        (folder / 'completion.json').write_text('{"completed":false}')
        return Mock(returncode=0)
    with patch('env_factory_v2.authoring.author_session.subprocess.Popen', side_effect=process):
        with pytest.raises(AuthorDraftError) as error:
            generate_in_session({'goal': 'test'}, evidence_root=tmp_path)
    assert error.value.recovery['previous_source'] == draft
    assert error.value.recovery['latest_operation']['result']['passed'] is False


@pytest.mark.parametrize('case', ['partial', 'checked', 'missing', 'unchanged', 'inflight', 'forbidden', 'truncated'])
def test_interrupted_author_recovers_only_audited_progress(tmp_path, case):
    from pathlib import Path
    import subprocess
    from unittest.mock import Mock
    from env_factory_v2.authoring.author_session import AuthorDraftError, generate_in_session

    draft = direct() if case == 'checked' else {'environment': {'version': '1.0', 'id': 'partial'}}
    previous = draft if case == 'unchanged' else None
    def process(args, **kwargs):
        folder = Path(args[args.index('--output-last-message') + 1]).parent
        if case != 'missing':
            server = AuthorMCP(folder / 'workbench', source=draft)
            server.dispatch({'id': 1, 'method': 'tools/call',
                'params': {'name': 'check_contract', 'arguments': {}}})
        event = {'type': 'item.completed', 'item': {'type': 'mcp_tool_call',
            'server': 'envfactory_author', 'tool': 'put_section', 'status': 'completed'}}
        if case == 'inflight':
            event['type'] = 'item.started'; event['item']['status'] = 'in_progress'
        events = [event]
        if case == 'partial':
            events.append({'type': 'item.completed', 'item': {'type': 'mcp_tool_call',
                'server': 'envfactory_author', 'tool': 'read_contract', 'status': 'completed'}})
        if case == 'forbidden': events.append({'type': 'item.started', 'item': {'type': 'command_execution'}})
        (folder / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n'
                                            + ('{"type":' if case == 'truncated' else ''))
        return Mock(pid=123, returncode=-15, communicate=Mock(side_effect=subprocess.TimeoutExpired('codex', 1)))
    with patch('env_factory_v2.authoring.author_session.subprocess.Popen', side_effect=process), \
         patch('env_factory_v2.authoring.author_session.os.killpg') as kill:
        expected = AuthorDraftError if case in {'partial', 'checked'} else (ContractError if case == 'forbidden'
                   else json.JSONDecodeError if case == 'truncated' else RuntimeError)
        with pytest.raises(expected) as error:
            generate_in_session({'goal': 'test'}, evidence_root=tmp_path, timeout=1, previous_source=previous)
        kill.assert_called_once()
    if case in {'partial', 'checked'}:
        assert error.value.recovery['previous_source'] == draft
        assert error.value.recovery['interrupted'] is True
        assert error.value.recovery['scope'] == 'unqualified_author_draft'
        # Even a successful authored check is only a draft when the turn times out.
        assert error.value.recovery['latest_operation']['result']['passed'] is (case == 'checked')
    else:
        assert not isinstance(error.value, AuthorDraftError)


def test_interrupted_draft_does_not_increase_pipeline_repair_budget(tmp_path):
    from unittest.mock import Mock
    from env_factory_v2.authoring.author_session import AuthorDraftError
    from env_factory_v2.pipeline import build
    draft = {'environment': {'version': '1.0', 'id': 'partial'}}
    agent = Mock()
    agent.generate.side_effect = AuthorDraftError('author session deadline exceeded', {
        'previous_source': draft, 'scope': 'unqualified_author_draft', 'interrupted': True})
    result = build({'goal': 'test'}, agent=agent, output=tmp_path / 'build', repairs=1, author_tools=True)
    assert agent.generate.call_count == 2 and len(result['attempts']) == 2
    assert not result['qualified']
    assert all(a['failed_phase'] == 'generation' for a in result['attempts'])


def test_production_generation_defaults_to_workbench_and_never_falls_back(tmp_path):
    from unittest.mock import Mock
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a'*24,
                            timeout=255, evidence_root=tmp_path)
    agent.ask = Mock(side_effect=AssertionError('legacy transport reached'))
    with patch('env_factory_v2.authoring.author_session.generate_in_session',
               side_effect=RuntimeError('workbench unavailable')) as run:
        with pytest.raises(RuntimeError, match='workbench unavailable'):
            agent.generate({'goal':'test'})
    assert run.call_count == 1
    agent.ask.assert_not_called()
    with pytest.raises(ContractError, match='workbench'):
        agent.generate({'goal':'test'}, author_tools=False)


def test_production_build_defaults_to_workbench(tmp_path):
    from unittest.mock import Mock
    from env_factory_v2.pipeline import build
    agent = Mock()
    agent.generate.side_effect = RuntimeError('stop after route selection')
    build({'goal':'test'},agent=agent,output=tmp_path/'run',repairs=0)
    assert agent.generate.call_args.kwargs['author_tools'] is True
    with pytest.raises(ContractError, match='workbench'):
        build({'goal':'test'},agent=agent,output=tmp_path/'legacy',author_tools=False)
    assert not (tmp_path/'legacy').exists()
