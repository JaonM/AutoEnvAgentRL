import json

from env_factory.generation.agent_cli import transport_diagnostics


def test_transport_notices_are_counted_without_guessing_elapsed_time():
    events = [
        {'type': 'error', 'message': 'Reconnecting... 2/5 (request timed out)'},
        {'type': 'item.completed', 'item': {'type': 'error',
          'message': 'Falling back from WebSockets to HTTPS transport. request timed out'}},
        # Agent-authored output quoting a diagnostic is not transport evidence.
        {'type': 'item.completed', 'item': {'type': 'command_execution',
          'aggregated_output': 'Reconnecting... 3/5'}},
        {'type': 'turn.completed', 'usage': {}},
    ]
    report = transport_diagnostics('\n'.join(map(json.dumps, events)) + '\n{"type":')
    assert report['reconnect_notices'] == 1
    assert report['https_fallback_notices'] == 1
    assert report['malformed_event_lines'] == 1
    assert report['network_seconds'] is None


def test_other_agent_events_and_empty_stream_do_not_claim_network_failure():
    for stream in ('', 'null\n[]\n', '{"type":"result","subtype":"success"}'):
        report = transport_diagnostics(stream)
        assert report['reconnect_notices'] == report['https_fallback_notices'] == 0
