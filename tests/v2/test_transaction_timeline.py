import copy

import pytest

from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import transaction, lit
from env_factory_v2.contracts.expressions import ContractError
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.contracts.transaction_targets import matches_plan, project_plan
from env_factory_v2.validation.verification import run_scenario


def temporal_source():
    source = transaction()
    source['tables'][0]['schema']['properties']['updated_at'] = {'type': 'integer'}
    for row in source['tables'][0]['rows']:
        row['updated_at'] = 0
    source['answers'][0]['schema']['items']['properties']['updated_at'] = {'type': 'integer'}
    source['tools'][1]['operations'][0]['set']['updated_at'] = {'ref': 'time.elapsed_seconds'}
    source['goals'] = [{'id': 'timed_transfer', 'rule_ids': ['nonnegative', 'conservation'],
        'transaction_timeline': [{'tool': 'transfer', 'arguments': {
            'from_id': lit('A'), 'to_id': lit('B'), 'amount': lit(20)}}]}]
    source['scenarios'][0]['steps'].append({'kind': 'wait', 'seconds': 4})
    return source


def test_projection_uses_commit_clock_not_episode_end():
    source = temporal_source()
    run = run_scenario(compile_contract(source), source['scenarios'][0])
    assert run['outcome']['success']
    assert run['final_state']['accounts'][0]['updated_at'] == 3
    assert run['time']['elapsed_seconds'] == 7


def test_events_at_commit_boundary_apply_before_transaction_and_later_events_after():
    source = temporal_source()
    source['events'] = [{'at_seconds': 3, 'table': 'accounts', 'key': 'A', 'set': {'updated_at': 80}}]
    first = run_scenario(compile_contract(source), source['scenarios'][0])
    assert first['outcome']['success']
    assert first['final_state']['accounts'][0]['updated_at'] == 3
    source['events'].append({'at_seconds': 5, 'table': 'accounts', 'key': 'A', 'set': {'updated_at': 90}})
    second = run_scenario(compile_contract(source), source['scenarios'][0])
    assert second['outcome']['success']
    assert second['final_state']['accounts'][0]['updated_at'] == 90


def test_forged_commit_results_do_not_change_projection_truth():
    source = temporal_source()
    compiled = compile_contract(source)
    run = run_scenario(compiled, source['scenarios'][0])
    context = Sandbox(compiled).context(private=True)
    context.update(data=run['final_state'], time=run['time'])
    trace = copy.deepcopy(run['trace'])
    commit = next(turn for turn in trace if turn.get('tool') == 'transfer')
    commit['result'] = {'committed': True, 'changes': 'untrusted summary'}
    plan = source['goals'][0]['transaction_timeline']
    assert matches_plan(plan, source, context, trace=trace)
    context['data']['accounts'][0]['updated_at'] = 7
    assert not matches_plan(plan, source, context, trace=trace)


@pytest.mark.parametrize('change', ['missing', 'extra', 'wrong_args'])
def test_timeline_cannot_replace_intention_with_actual_behavior(change):
    source = temporal_source()
    compiled = compile_contract(source)
    run = run_scenario(compiled, source['scenarios'][0])
    context = Sandbox(compiled).context(private=True)
    context.update(data=run['final_state'], time=run['time'])
    trace = copy.deepcopy(run['trace'])
    commit = next(turn for turn in trace if turn.get('tool') == 'transfer')
    if change == 'missing':
        trace.remove(commit)
    elif change == 'extra':
        trace.append(copy.deepcopy(commit))
    else:
        commit['arguments']['amount'] = 19
    with pytest.raises(ContractError):
        project_plan(source['goals'][0]['transaction_timeline'], source, context, trace=trace)


def test_disabled_intention_still_projects_events_without_requiring_commit():
    source = temporal_source()
    source['goals'][0]['transaction_timeline'][0]['when'] = lit(False)
    source['events'] = [{'at_seconds': 0, 'table': 'accounts', 'key': 'A', 'set': {'updated_at': 2}}]
    sandbox = Sandbox(compile_contract(source))
    assert matches_plan(source['goals'][0]['transaction_timeline'], source,
                        sandbox.context(private=True), trace=[])
