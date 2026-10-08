import json
from unittest.mock import Mock

import pytest

from env_factory.sandbox_runtime import ContractUserSimulator, EpisodeStore, SandboxError


def stage():
    return {'id': 'size', 'kind': 'information_required', 'before_tool': 'find_sets',
        'bind_to': {'argument': 'size'}, 'private_fact': 37,
        'user_reply': 'My size is 37.', 'retry_reply': 'Verify the requested event first.',
        'requires_tools': ['lookup'], 'requires_evidence': [{
            'tool': 'lookup', 'arguments': {'name': 'Autumn event'},
            'conditions': [{'path': '$.count', 'operator': 'eq', 'expected': 1}]}]}


def event(name='Autumn event', count=1, kind='tool_call'):
    return {'event': kind, 'payload': {'tool_name': 'lookup', 'arguments': {'name': name}},
            'result': {'count': count, 'records': [{'id': 'E'}] if count else []}}


def turn(tmp_path, events, declaration=None):
    sim = object.__new__(ContractUserSimulator)
    sim.episode_store = EpisodeStore(tmp_path / 'episodes.sqlite3')
    sim.episode_store.reset(seed=17)
    sim._execution_route = Mock(return_value=None)
    for item in events:
        sim.episode_store.event(item['event'], item['payload'], item['result'])
    script = {'interaction_protocol': {'request_mode': 'structured_v1',
                                      'stages': [declaration or stage()]}}
    question = json.dumps({'question': 'What is your size?', 'request': {
        'intent': 'request_input', 'tool': 'find_sets', 'arguments': ['size']}})
    result = sim._protocol_turn(script, {'script_id': 'test', 'turn_index': 0},
                               [{'role': 'assistant', 'content': question}])
    return result, sim.episode_store.get_state(sim.STATE_KEY)


@pytest.mark.parametrize('events', [[], [event(count=0)], [event(count=2)],
    [event(name='Other event')], [event(), event(name='Other event')],
    [event(), event(count=0)], [event(count=True)]])
def test_disclosure_requires_current_matching_unique_evidence(tmp_path, events):
    result, state = turn(tmp_path, events)
    assert not result['fsm_transition_applied']
    assert '37' not in result['user_query']
    assert state['protocol_index'] == 0


@pytest.mark.parametrize('events', [[event()], [event(name='Other event'), event()],
    [event(kind='tool_call_replay')]])
def test_correct_evidence_and_recovery_allow_disclosure(tmp_path, events):
    result, state = turn(tmp_path, events)
    assert result['fsm_transition_applied'] and '37' in result['user_query']
    assert state['protocol_index'] == 1


@pytest.mark.parametrize('bad', [None, [], [{'tool': 'lookup', 'conditions': []}],
    [{'tool': 'undeclared', 'conditions': [{'path': '$.count', 'operator': 'eq', 'expected': 1}]}],
    [{'tool': 'lookup', 'conditions': [{'path': '$.count', 'operator': 'invented', 'expected': 1}]}]])
def test_malformed_declared_evidence_fails_closed(tmp_path, bad):
    declared = stage()
    declared['requires_evidence'] = bad
    with pytest.raises(SandboxError) as error:
        turn(tmp_path, [event()], declared)
    assert error.value.code == 'INTERACTION_EVIDENCE_INVALID'
