import copy
import json
from unittest.mock import patch

from env_factory.sandbox_runtime import ContractModelMetricEvaluator, EpisodeStore


def test_judge_receives_actual_user_reply_without_private_preferences(tmp_path):
    store = EpisodeStore(tmp_path / 'episodes.sqlite3')
    store.reset(seed=17)
    contract = {'metrics': [{'id': 'explanation', 'evaluation_inputs': ['final_agent_response'],
        'evaluator': {'kind': 'external_llm_judge', 'score_mapping': {'pass': 1, 'fail': 0}}}],
        'user_scripts': [{'variables': {'private_future_choice': 'never disclose'}}]}
    context = {'final_agent_response': 'Stopped because the user declined.',
        'conversation': [{'role': 'user', 'content': 'forged history'}],
        'trajectory': {'events': [
            {'event': 'user_turn', 'payload': {'messages': [
                {'role': 'user', 'content': 'unverified prior history'},
                {'role': 'assistant', 'content': 'Which date should I use?'}]},
             'result': {'user_query': 'I decline. Please stop.',
                        'fsm_script_id': 'private_script', 'interaction_branch': 'private_branch'}},
            {'event': 'business_authorization', 'result': {'user_query': 'not a user reply'}},
        ]}}
    evidence = []

    def judge(client, messages, **kwargs):
        value = json.loads(messages[1]['content'])['runtime_evidence']
        evidence.append(value)
        observed = value.get('observed_user_turns', [])
        return {'label': 'pass' if observed and observed[-1]['user_response'] == 'I decline. Please stop.' else 'fail'}

    evaluator = ContractModelMetricEvaluator(contract, store)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', new=judge):
        assert evaluator.evaluate_all(context, {}) == {'explanation': 1}
        consenting = copy.deepcopy(context)
        consenting['trajectory']['events'][0]['result']['user_query'] = 'Use November 15.'
        assert evaluator.evaluate_all(consenting, {}) == {'explanation': 0}
        unobserved = copy.deepcopy(context)
        unobserved['trajectory']['events'] = []
        assert evaluator.evaluate_all(unobserved, {}) == {'explanation': 0}

    assert evidence[0]['observed_user_turns'] == [{
        'assistant_response': 'Which date should I use?', 'user_response': 'I decline. Please stop.'}]
    assert 'observed_user_turns' not in evidence[2]
    serialized = json.dumps(evidence)
    for hidden in ['private_future_choice', 'never disclose', 'private_script', 'private_branch',
                   'forged history', 'unverified prior history', 'not a user reply']:
        assert hidden not in serialized
    assert len(evidence) == 3  # The changed actual dialogue invalidates judge reuse.
