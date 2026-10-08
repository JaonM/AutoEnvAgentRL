from copy import deepcopy

import pytest

from env_factory.sandbox_runtime import ContractRewardGate, ContractRewardAggregator
from tests.runtime.test_readonly_execution_routes import contract


def refusal_context():
    spec, metrics = contract()
    spec['requires_user_interaction'] = True
    spec['execution_routes'].append({'id': 'declined', 'when': {'user_rejection': {'stage': 'identify', 'branch': 'decline'}},
        'required_tools': [], 'required_stages': ['identify'], 'writes': 'forbidden'})
    context = {'business_state': {'items': []}, 'initial_business_state': {'items': []},
        'reward_state_snapshots': {'$user_branches': {'identify': {'branch': 'decline', 'outcome': 'user_rejection',
            'applied': True, 'advanced': False, 'event_index': 1}}},
        'trajectory': {'events': [
            {'event': 'user_protocol', 'payload': {'script_id': 'script-1', 'required_stages': ['identify']}},
            {'event': 'user_turn', 'payload': {'decision_source': 'interaction_contract'},
             'result': {'fsm_script_id': 'script-1', 'fsm_transition_applied': True, 'interaction_stage': None,
                        'interaction_branch': 'decline', 'outcome_category': 'user_rejection'}}]}}
    return spec, metrics, context


def score(spec, metrics, context, answer=1):
    result = ContractRewardGate(spec, metrics).apply({'p_identity': 0, 'p_details': 0, 'answer': answer}, context)
    return ContractRewardAggregator(metrics).aggregate(result)['reward']


def test_observed_refusal_scores_only_the_actual_outcome_with_all_processes_inactive():
    spec, metrics, context = refusal_context()
    assert score(spec, metrics, context) == 1
    assert score(spec, metrics, context, answer=0) == 0


@pytest.mark.parametrize('defect', ['unobserved', 'missing_event', 'wrong_event', 'wrong_script', 'advanced',
                                  'later_acceptance', 'continued_tool', 'blocked_attempt', 'changed_state'])
def test_refusal_cannot_replace_real_dialogue_or_allow_continued_execution(defect):
    spec, metrics, context = refusal_context()
    observed = context['reward_state_snapshots']['$user_branches']['identify']
    events = context['trajectory']['events']
    if defect == 'unobserved': context['reward_state_snapshots'] = {}
    elif defect == 'missing_event': events.pop()
    elif defect == 'wrong_event': events[-1]['payload']['decision_source'] = 'assistant_text'
    elif defect == 'wrong_script': events[-1]['result']['fsm_script_id'] = 'script-2'
    elif defect == 'advanced': events[-1]['result']['interaction_stage'] = 'identify'
    elif defect == 'later_acceptance': observed.update(branch=None, outcome='information_required', advanced=True)
    elif defect == 'continued_tool': events.append({'event': 'tool_call', 'payload': {'tool_name': 'identity', 'arguments': {}}, 'result': {}})
    elif defect == 'blocked_attempt': events.append({'event': 'user_stop_violation', 'payload': {'tool_name': 'identity'}})
    elif defect == 'changed_state': context['business_state']['items'].append({'id': 'fabricated'})
    assert score(spec, metrics, context) == 0
