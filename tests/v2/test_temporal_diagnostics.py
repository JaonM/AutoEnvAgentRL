import copy
from unittest.mock import Mock

import pytest

from env_factory_v2.contracts.contract import compile_contract, ContractError
from env_factory_v2.examples import transaction, lit, ref
from env_factory_v2.validation.temporal_diagnostics import temporal_diagnostics
from env_factory_v2.authoring.author_workbench import AuthorWorkbench


def source():
    s=transaction()
    s['goals'].append({'id':'clock_resolution','rule_ids':['conservation'],
        'predicate':{'ge':[ref('time.elapsed_seconds'),lit(50)]}})
    return s


def test_wait_only_transition_is_evidence_not_a_verdict_or_mutation():
    c=compile_contract(source()); original=copy.deepcopy(c)
    r=temporal_diagnostics(c)
    f=next(x for x in r['findings'] if x['wait_seconds']==50)
    assert f['goal']=='clock_resolution' and f['same_world_and_user_state']
    assert not f['full_reward_tested'] and 'passed' not in r
    assert c==original


def test_scheduled_state_change_is_not_called_clock_only():
    s=source(); s['events']=[{'at_seconds':50,'table':'accounts','key':'A','set':{'balance':5}}]
    r=temporal_diagnostics(compile_contract(s))
    assert r['changed_world_excluded'] and not r['findings']


def test_no_clock_transition_no_hint():
    assert not temporal_diagnostics(compile_contract(transaction()))['findings']


def test_budget_can_disable_or_bound_probe_count():
    c=compile_contract(source())
    assert not temporal_diagnostics(c,limit=0)['findings']
    assert len(temporal_diagnostics(c,limit=1)['candidate_seconds'])==1
    with pytest.raises(ContractError): temporal_diagnostics(c,limit=17)


def test_malformed_goal_evaluation_does_not_become_false_business_verdict():
    s=source(); s['goals'][-1]['predicate']={'eq':[ref('data.missing'),lit(1)]}
    assert not temporal_diagnostics(compile_contract(s))['findings']


def test_legitimate_wait_hint_does_not_change_recipe_check_verdict():
    s=source()
    for scenario in s['scenarios']:
        scenario['steps'].append({'kind':'wait','seconds':50})
    result=AuthorWorkbench(source=s).call('check_contract',{})
    assert result['passed'] and result['temporal_diagnostics']['findings']
