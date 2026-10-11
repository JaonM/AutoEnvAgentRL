"""Behavioral statechart checks, including invalidation and authorization safety."""
import copy
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from env_factory_v2.examples import transaction
from env_factory_v2.contracts.expressions import ContractError
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.execution.user_simulator import UserSimulator
from env_factory_v2.delivery.server import serve


def consent_world(source=None):
    world = Sandbox(source or transaction())
    world.call('accounts', {}, call_id='balances')
    args = {'from_id': {'$result': 'balances', 'path': 'items.0.id'},
            'to_id': {'$result': 'balances', 'path': 'items.1.id'}, 'amount': 20}
    proposal = {'tool': 'transfer', 'arguments': args}
    world.interact('approve', question='批准转账？', proposal=proposal)
    return world, args, proposal


def test_revocation_requires_fresh_confirmation_and_consumption_is_terminal():
    world, args, proposal = consent_world()
    world.revoke('approve')
    assert world.user.states['approve'] == 'consent.revoked'
    assert 'approve' not in world.user.completed
    world.interact('approve', question='重新确认？', proposal=proposal)
    world.call('transfer', args, call_id='write')
    assert world.user.states['approve'] == 'consent.consumed'
    with pytest.raises(ContractError, match='consumed'):
        world.interact('approve', question='再执行？', proposal=proposal)
    assert [row['balance'] for row in world.data['accounts']] == [80, 30]


def test_changed_attempt_invalidates_old_consent_without_restoring_it():
    world, args, proposal = consent_world()
    altered = {**args, 'from_id': args['to_id'], 'to_id': args['from_id']}
    with pytest.raises(ContractError, match='parameters changed'):
        world.call('transfer', altered, call_id='altered')
    assert world.user.states['approve'] == 'consent.invalidated'
    with pytest.raises(ContractError, match='authorization missing'):
        world.user.authorize('approve', 'transfer', world.resolve(args))
    world.interact('approve', question='确认原方案？', proposal=proposal)
    assert world.user.states['approve'] == 'consent.authorized'


def test_changed_proposal_requires_another_actual_user_turn():
    world, args, _ = consent_world()
    altered = {**args, 'from_id': args['to_id'], 'to_id': args['from_id']}
    world.interact('approve', question='改为反向转账？', proposal={'tool': 'transfer', 'arguments': altered})
    assert world.user.completed['approve']['proposal']['arguments']['from_id'] == world.data['accounts'][1]['id']
    assert [h['event'] for h in world.user.history][-3:] == ['invalidate', 'propose', 'approve']
    world.user.authorize('approve', 'transfer', world.resolve(altered))


def test_failed_atomic_execution_retains_consent_for_retry():
    source = transaction()
    source['tools'][1]['operations'][1]['key'] = {'literal': 'absent'}
    world, args, _ = consent_world(source)
    before = copy.deepcopy(world.data)
    with pytest.raises(ContractError, match='exactly one row'):
        world.call('transfer', args, call_id='failed')
    assert world.data == before
    assert world.user.states['approve'] == 'consent.authorized'
    world.user.authorize('approve', 'transfer', world.resolve(args))


def test_independent_regions_and_dependency_guards():
    contract = {'user': {'facts': {'a': 1, 'b': 2, 'c': 3}, 'stages': [
        {'id': 'a', 'kind': 'fact', 'fact': 'a', 'depends_on': []},
        {'id': 'b', 'kind': 'fact', 'fact': 'b', 'depends_on': []},
        {'id': 'c', 'kind': 'fact', 'fact': 'c', 'depends_on': ['a']}]}}
    user = UserSimulator(contract)
    assert user.states['c'] == 'waiting.blocked'
    with pytest.raises(ContractError, match='prerequisites'):
        user.respond('c', observations={}, tools={})
    user.respond('b', observations={}, tools={})
    assert user.states['a'] == 'waiting.ready'
    user.respond('a', observations={}, tools={})
    assert user.states['c'] == 'waiting.ready'
    user.respond('c', observations={}, tools={})
    with pytest.raises(ContractError, match='already completed'):
        user.respond('c', observations={}, tools={})


def test_fact_correction_invalidates_consent_and_dependent_choices():
    source = transaction()
    source['user']['facts'].update({'size': 'M', 'corrected': 'L'})
    source['user']['stages'] = [
        {'id': 'size', 'kind': 'fact', 'fact': 'size', 'depends_on': []},
        {**source['user']['stages'][0], 'depends_on': ['size']},
        {'id': 'choice', 'kind': 'choice', 'fact': 'size', 'as_fact': 'chosen',
         'source_tool': 'menu', 'source_path': 'items', 'value_field': 'id',
         'fallback': 'first_available', 'depends_on': ['size']},
        {'id': 'correction', 'kind': 'correction', 'fact': 'corrected', 'as_fact': 'size',
         'depends_on': ['size', 'approve']}]
    user = UserSimulator(source)
    tools = {t['id']: t for t in source['tools']}
    menu = {'m': {'tool': 'menu', 'result': {'items': [{'id': 'M'}, {'id': 'L'}]}}}
    user.respond('size', observations=menu, tools=tools)
    user.respond('choice', observations=menu, tools=tools, source_id='m')
    proposal = {'tool': 'transfer', 'arguments': {'from_id': 'A', 'to_id': 'B', 'amount': 20}}
    user.respond('approve', observations=menu, tools=tools, proposal=proposal)
    user.respond('correction', observations=menu, tools=tools)
    assert user.states['approve'] == 'consent.invalidated'
    assert user.states['choice'] == 'waiting.ready'
    assert 'chosen' not in user.disclosed
    assert user.disclosed['size'] == 'L'
    assert user.states['correction'] == 'resolved.disclosed'
    assert user.respond('choice', observations=menu, tools=tools, source_id='m')['value'] == 'L'
    with pytest.raises(ContractError, match='authorization missing'):
        user.authorize('approve', 'transfer', proposal['arguments'])


def test_declined_and_no_menu_choices_are_explicit_states():
    source = transaction()
    source['user']['facts']['approve_approve'] = False
    user = UserSimulator(source)
    proposal = {'tool': 'transfer', 'arguments': {'from_id': 'A', 'to_id': 'B', 'amount': 20}}
    result = user.respond('approve', observations={}, tools={t['id']: t for t in source['tools']}, proposal=proposal)
    assert result['approved'] is False
    assert user.states['approve'] == 'consent.declined'
    with pytest.raises(ContractError, match='authorization missing'):
        user.authorize('approve', 'transfer', proposal['arguments'])
    source['user'] = {'facts': {'preference': 'secret'}, 'stages': [
        {'id': 'pick', 'kind': 'choice', 'fact': 'preference', 'depends_on': [],
         'source_tool': 'menu', 'source_path': 'items', 'value_field': 'id', 'fallback': 'refuse'}]}
    user = UserSimulator(source)
    result = user.respond('pick', observations={'m': {'tool': 'menu', 'result': {'items': []}}}, tools={}, source_id='m')
    assert result['refused'] is True
    assert user.states['pick'] == 'resolved.refused'
    assert 'secret' not in json.dumps(user.snapshot())


def test_invalid_proposal_does_not_change_state_and_reset_clears_history():
    world, _, _ = consent_world()
    before = world.user.snapshot()
    with pytest.raises(ContractError):
        world.interact('approve', question='错误方案', proposal={'tool': 'accounts', 'arguments': {}})
    assert world.user.snapshot() == before
    before['regions']['approve'] = 'tampered'
    assert world.user.states['approve'] == 'consent.authorized'
    world.reset()
    assert world.user.states['approve'] == 'waiting.ready'
    assert world.user.history == [] and world.user.authorizations == {}


def test_http_state_read_is_episode_bound_private_and_does_not_advance_time():
    server = serve(transaction(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def request(route, body=None, episode=None):
        headers = {'X-Episode': episode} if episode else {}
        data = json.dumps(body).encode() if body is not None else None
        with urlopen(Request(base + route, data=data, headers=headers), timeout=5) as response:
            return json.load(response)
    try:
        reset = request('/reset', {})
        episode = reset['episode']
        assert reset['public']['user_statechart']['consumed_is_terminal'] is True
        with pytest.raises(HTTPError) as error:
            request('/user-state')
        assert error.value.code == 404
        before = request('/time', episode=episode)
        state = request('/user-state', episode=episode)
        assert state['regions']['approve'] == 'waiting.ready'
        assert set(state) == {'root', 'regions', 'history'}
        assert request('/time', episode=episode) == before
        request('/tool', {'tool': 'accounts', 'arguments': {}, 'id': 'read'}, episode)
        args = {'from_id': {'$result': 'read', 'path': 'items.0.id'},
                'to_id': {'$result': 'read', 'path': 'items.1.id'}, 'amount': 20}
        request('/user', {'stage': 'approve', 'question': '确认？',
                         'proposal': {'tool': 'transfer', 'arguments': args}}, episode)
        state = request('/user-state', episode=episode)
        assert state['regions']['approve'] == 'consent.authorized'
        assert 'arguments' not in json.dumps(state) and 'approve_approve' not in json.dumps(state)
        other = request('/reset', {})['episode']
        assert request('/user-state', episode=other)['regions']['approve'] == 'waiting.ready'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
