import json
import subprocess
import sys

from env_factory_v2.authoring.author_mcp import AuthorMCP
from env_factory_v2.examples import direct


def call(server, name, arguments):
    response = server.dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                                'params': {'name': name, 'arguments': arguments}})
    return json.loads(response['result']['content'][0]['text'])


def test_receipt_tracks_current_draft_and_never_reuses_stale_pass(tmp_path):
    server = AuthorMCP(tmp_path, source=direct())
    assert call(server, 'check_contract', {})['passed']
    first = json.loads((tmp_path/'latest_operation.json').read_text())
    full = json.loads((tmp_path/'check-0001.json').read_text())
    assert full['source_identity'] == first['result']['source_identity']
    assert len(full['result']['checks']) == first['result']['check_count']
    assert call(server, 'put_section', {'section': 'answers', 'value_json': '[]'})['passed']
    latest = json.loads((tmp_path/'latest_operation.json').read_text())
    assert latest['tool'] == 'put_section'
    assert latest['result']['source_identity'] != first['result']['source_identity']
    assert not call(server, 'check_contract', {})['passed']
    assert len((tmp_path/'operations.jsonl').read_text().splitlines()) == 3


def test_stdio_handshake_tool_discovery_and_real_check(tmp_path):
    source = tmp_path/'initial.json'
    source.write_text(json.dumps(direct()))
    requests = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05'}},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'check_contract', 'arguments': {}}},
    ]
    process = subprocess.run([sys.executable, '-m', 'env_factory_v2.authoring.author_mcp', '--output', str(tmp_path/'session'),
                              '--source', str(source)], input='\n'.join(map(json.dumps, requests))+'\n',
                             text=True, capture_output=True, check=True, timeout=10)
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(responses) == 3
    assert {t['name'] for t in responses[1]['result']['tools']} == {'read_contract', 'inspect_contract', 'read_check', 'read_context', 'put_section', 'edit_contract', 'check_contract', 'submit_contract'}
    assert json.loads(responses[2]['result']['content'][0]['text'])['passed']


def test_model_cannot_select_files_or_execute_commands(tmp_path):
    server = AuthorMCP(tmp_path, source=direct())
    assert not call(server, 'shell', {'cmd': 'anything'})['passed']
    assert not call(server, 'put_section', {'section': '../elsewhere', 'value_json': '{}'})['passed']
    assert not call(server, 'check_contract', {'path': '../elsewhere'})['passed']
    assert json.loads((tmp_path/'draft.json').read_text()) == direct()


def test_read_preserves_current_check_and_cannot_create_a_check(tmp_path):
    from env_factory_v2.authoring.author_session import checked_source
    folder = tmp_path / 'workbench'
    server = AuthorMCP(folder, source=direct())
    response = call(server, 'read_contract', {'path': '/task', 'offset': 0})
    assert response['passed'] and json.loads(response['json_fragment']) == direct()['task']
    assert not (folder/'latest_operation.json').exists()
    assert call(server, 'check_contract', {})['passed']
    receipt = (folder/'latest_operation.json').read_bytes()
    assert call(server, 'read_contract', {'path': '', 'offset': 0})['passed']
    assert not call(server, 'read_contract', {'path': '/missing', 'offset': 0})['passed']
    assert (folder/'latest_operation.json').read_bytes() == receipt
    assert checked_source(tmp_path)[0] == direct()
    assert len((folder/'operations.jsonl').read_text().splitlines()) == 4


def test_mcp_advertises_only_the_selected_author_format(tmp_path):
    fresh=AuthorMCP(tmp_path/'fresh')
    legacy=AuthorMCP(tmp_path/'legacy',source=direct())
    def sections(server):
        value=server.dispatch({'id':1,'method':'tools/list'})['result']['tools']
        return next(t for t in value if t['name']=='put_section')['inputSchema']['properties']['section']['enum']
    assert sections(fresh)==['environment','task']
    assert 'rules' in sections(legacy) and 'environment' not in sections(legacy)
    assert sections(fresh)==['environment','task']
