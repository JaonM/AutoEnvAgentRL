import copy

import pytest

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import transaction
from env_factory_v2.expressions import ContractError
from env_factory_v2.verification import authored_cases, run_scenario, replay_identity


def test_private_decline_variant_blocks_write_and_preserves_base_fixture():
    source = transaction()
    original = copy.deepcopy(source)
    source['scenarios'].append({'id': 'declined_write', 'extends': 'correct',
        'expect_success': False, 'expect_error': 'authorization missing',
        'user_overrides': {'approve_approve': False}})
    compiled = compile_contract(source)
    variant = compiled['source']['scenarios'][-1]
    first = run_scenario(compiled, variant)
    assert 'authorization missing' in first['error']
    assert first['final_state']['accounts'] == original['tables'][0]['rows']
    assert replay_identity(first) == replay_identity(run_scenario(compiled, variant))
    assert compiled['source']['user']['facts'] == original['user']['facts']
    assert run_scenario(compiled, compiled['source']['scenarios'][0])['outcome']['success']


@pytest.mark.parametrize('overrides', [{'unknown': False}, {'approve_approve': 0}, {'approve_approve': 'false'}])
def test_invalid_private_intent_override_rejected(overrides):
    source = transaction()
    source['scenarios'][1]['user_overrides'] = overrides
    with pytest.raises(ContractError):
        compile_contract(source)


def test_probe_overrides_materialize_without_mutating_parent():
    source = transaction()
    source['probes'][0]['user_overrides'] = {'approve_approve': False}
    compiled = compile_contract(source)
    cases = list(authored_cases(compiled))
    probe = cases[-1][1]
    assert probe['user_overrides'] == {'approve_approve': False}
    assert 'user_overrides' not in compiled['source']['scenarios'][0]


def test_independent_challenge_executes_private_decline_and_rejects_unknown_key():
    import json
    from types import SimpleNamespace
    from env_factory_v2.challenge import _execute_case

    compiled = compile_contract(transaction())
    scenarios = {s['id']: s for s in compiled['source']['scenarios']}
    agent = SimpleNamespace(judge=None, independent_judge=None)
    case = {'id': 'independent_decline', 'scenario_id': 'correct',
            'mutations_json': '[]', 'answer_json': '{}', 'expect_success': False,
            'user_overrides_json': json.dumps({'approve_approve': False})}
    result = _execute_case(compiled, case, scenarios, agent)
    assert not result['passed']  # Unexpected fixture errors cannot prove a negative.
    assert 'authorization missing' in result['first']['error']
    assert result['first']['final_state']['accounts'] == compiled['source']['tables'][0]['rows']
    case['user_overrides_json'] = '{"unknown":false}'
    with pytest.raises(ContractError, match='unknown private user override'):
        _execute_case(compiled, case, scenarios, agent)
