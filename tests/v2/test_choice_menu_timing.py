import pytest

from env_factory_v2.examples import transaction
from env_factory_v2.expressions import ContractError
from env_factory_v2.runtime import Sandbox


def menu_world(*, complete=True, events=False):
    source = transaction()
    source['tools'][0].update(page_size=1, complete_retrieval=complete)
    source['user']['facts']['preferred'] = 'A'
    source['user']['stages'].append({'id': 'choose', 'kind': 'choice', 'fact': 'preferred',
        'source_tool': 'accounts', 'source_path': 'items', 'value_field': 'id',
        'fallback': 'first_available', 'depends_on': []})
    if events:
        source['time']['seconds_per_action'] = 0
        source['events'] = [{'at_seconds': 1, 'table': 'accounts', 'key': 'A', 'set': {'balance': 90}}]
    return Sandbox(source)


def test_choice_cannot_precede_complete_retrieval_or_use_just_one_page():
    world = menu_world()
    first = world.call('accounts', {}, call_id='first')
    with pytest.raises(ContractError, match='complete current observed menu before selection'):
        world.interact('choose', question='选择账户？', source_id='first')
    assert 'choose' not in world.user.completed
    world.call('accounts', {}, call_id='second', cursor=first['next_cursor'])
    # Traversal alone is insufficient: the user must receive the complete menu.
    with pytest.raises(ContractError, match='complete current observed menu before selection'):
        world.interact('choose', question='选择账户？', source_id='first')
    world.collect(['first', 'second'], call_id='menu')
    result = world.interact('choose', question='选择账户？', source_id='menu')
    assert result['value'] == 'A' and result['options'] == ['A', 'B']


def test_stale_complete_menu_cannot_be_used_after_an_event():
    world = menu_world(events=True)
    first = world.call('accounts', {}, call_id='first')
    world.call('accounts', {}, call_id='second', cursor=first['next_cursor'])
    world.collect(['first', 'second'], call_id='menu')
    world.wait(1)
    with pytest.raises(ContractError, match='complete current observed menu before selection'):
        world.interact('choose', question='选择账户？', source_id='menu')
    first = world.call('accounts', {}, call_id='new_first')
    world.call('accounts', {}, call_id='new_second', cursor=first['next_cursor'])
    world.collect(['new_first', 'new_second'], call_id='new_menu')
    assert world.interact('choose', question='选择账户？', source_id='new_menu')['value'] == 'A'


def test_partial_menu_choice_remains_valid_when_business_allows_it():
    world = menu_world(complete=False)
    world.call('accounts', {}, call_id='first')
    assert world.interact('choose', question='选择账户？', source_id='first')['options'] == ['A']
