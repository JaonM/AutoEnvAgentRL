import copy
import json
from unittest.mock import patch

from env_factory.sandbox_runtime import ContractEvaluatorRuntime, ContractModelMetricEvaluator, EpisodeStore


def test_each_semantic_metric_receives_its_own_field_scope_and_scores_separately(tmp_path):
    metrics = [{'id': name, 'category': 'outcome', 'scope': 'terminal', 'semantic_fields': [name],
        'evaluation_inputs': ['final_agent_response'], 'criteria': ['Truthful ' + name],
        'evaluator': {'kind': 'external_llm_judge', 'score_mapping': {'pass': 1, 'fail': 0}}}
        for name in ['analysis', 'recovery']]
    schema = {'type': 'object', 'properties': {name: {'type': 'string'} for name in ['analysis', 'recovery']},
              'required': ['analysis', 'recovery'], 'additionalProperties': False}
    contract = {'metrics': metrics, 'public_input': {'initial_user_message': 'Analyze evidence and report recovery.',
        'answer_contract': {'format': 'json_object', 'schema': schema}}}
    store = EpisodeStore(tmp_path/'state.sqlite3'); store.reset(seed=17)
    evaluator = ContractModelMetricEvaluator(contract, store)
    observed = []
    def judge(client, messages, **kwargs):
        instruction = json.loads(messages[0]['content'].split('schema. ', 1)[1])
        fields = instruction['metric_field_scope']['fields']
        evidence = json.loads(messages[1]['content'])['runtime_evidence']
        answer = json.loads(evidence['final_agent_response'])
        assert set(answer) == set(fields)
        observed.append(fields)
        return {'label': 'pass' if answer[fields[0]] == 'supported' else 'fail'}
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', new=judge):
        scores = evaluator.evaluate_all({'final_agent_response': json.dumps({
            'analysis': 'supported', 'recovery': 'invented'})}, {})
    assert scores == {'analysis': 1, 'recovery': 0}
    assert observed == [['analysis'], ['recovery']]


def test_field_scope_is_part_of_judgment_cache_identity(tmp_path):
    store = EpisodeStore(tmp_path/'cache.sqlite3'); store.reset(seed=17)
    evaluator = ContractEvaluatorRuntime(store)
    metric = {'id': 'quality', 'criteria': ['Grounded content'], 'semantic_fields': ['analysis'],
              'evaluator': {'score_mapping': {'pass': 1, 'fail': 0}}}
    context = {'final_agent_response': '{"analysis":"text"}'}
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as call:
        evaluator.json_judge(metric, context, fallback={'label': 'fail'})
        evaluator.json_judge(metric, context, fallback={'label': 'fail'})
        unscoped = copy.deepcopy(metric); unscoped.pop('semantic_fields')
        evaluator.json_judge(unscoped, context, fallback={'label': 'fail'})
    assert call.call_count == 2
