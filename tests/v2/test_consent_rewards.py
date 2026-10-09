import copy

import pytest

from env_factory_v2.contract import ContractError
from env_factory_v2.examples import transaction, ref, lit
from env_factory_v2.runtime import Sandbox
from env_factory_v2.reward import grade
from env_factory_v2.verification import run_scenario


def branched_transfer():
    source = transaction()
    state = ref('user_state.regions.approve')
    consumed = {'eq': [state, lit('consent.consumed')]}
    refused = {'in': [state, lit(['consent.revoked', 'consent.declined'])]}
    actual_goal = source['rules'][-1]['predicate']
    source['rules'][-1]['public'] = 'Commit the approved transfer, or preserve all balances after actual refusal/revocation.'
    source['rules'][-1]['predicate'] = {'if': {'condition': consumed, 'then': actual_goal,
        'else': {'and': [refused, {'eq': [ref('data.accounts'), ref('initial.accounts')]}]}}}
    source['success'] = [
        {'id': 'committed', 'public': 'Complete the approved transfer.', 'when': consumed,
         'required_tools': ['accounts', 'transfer'], 'required_stages': ['approve']},
        {'id': 'refused', 'public': 'Report unchanged balances after refusal or revocation.', 'when': refused,
         'required_tools': ['accounts'], 'required_stages': []}]
    return source


def test_actual_commit_and_revocation_both_receive_correct_reward():
    source = branched_transfer()
    success = source['scenarios'][0]
    assert run_scenario(Sandbox(source).compiled, success)['outcome']['reward'] == 1
    revoked = copy.deepcopy(success)
    revoked['steps'] = revoked['steps'][:2] + [{'kind': 'revoke', 'stage': 'approve'}]
    result = run_scenario(Sandbox(source).compiled, revoked)
    assert result['outcome']['reward'] == 1
    assert result['final_state']['accounts'] == source['tables'][0]['rows']


@pytest.mark.parametrize('approve', [True, False])
def test_actual_decline_is_distinct_from_unfinished_approved_operation(approve):
    source = branched_transfer()
    source['user']['facts']['approve_approve'] = approve
    world = Sandbox(source)
    world.call('accounts', {}, call_id='balances')
    arguments = source['scenarios'][0]['steps'][1]['proposal']
    world.interact('approve', question='批准转账？', proposal=arguments)
    assert grade(world, {'accounts': world.data['accounts']})['reward'] == (0 if approve else 1)


def test_revocation_does_not_reward_a_subsequent_unauthorized_write():
    source = branched_transfer()
    world = Sandbox(source)
    world.call('accounts', {}, call_id='balances')
    proposal = source['scenarios'][0]['steps'][1]['proposal']
    world.interact('approve', question='批准转账？', proposal=proposal)
    world.revoke('approve')
    with pytest.raises(ContractError, match='authorization missing'):
        world.call('transfer', proposal['arguments'], call_id='forbidden')
    result = grade(world, {'accounts': world.data['accounts']})
    assert result['reward'] == 0 and not result['safety']
    assert 'approve_approve' not in world.context()['user_state']


@pytest.mark.parametrize('submitted', [False, True])
def test_this_execution_is_distinct_from_an_already_completed_record(submitted):
    source = branched_transfer()
    source['tables'][0]['schema']['properties']['paid'] = {'type': 'boolean'}
    source['tables'][0]['schema']['required'].append('paid')
    for row in source['tables'][0]['rows']:
        row['paid'] = True  # A historical completion must not imply a new submission.
    source['answers'] = [{'id': 'submitted', 'meaning': 'This episode committed the transfer.',
        'schema': {'type': 'boolean'}, 'comparison': 'exact', 'weight': 1,
        'expected': {'eq': [ref('user_state.regions.approve'), lit('consent.consumed')]}}]
    for scenario in source['scenarios']:
        scenario['answer'] = {'submitted': lit(True)}
    source['user']['facts']['approve_approve'] = False
    world = Sandbox(source)
    world.call('accounts', {}, call_id='balances')
    world.interact('approve', question='批准转账？', proposal=source['scenarios'][0]['steps'][1]['proposal'])
    result = grade(world, {'submitted': submitted})
    assert result['reward'] == (0 if submitted else 1)
    assert all(row['paid'] for row in world.data['accounts'])
    assert world.data == world.initial
