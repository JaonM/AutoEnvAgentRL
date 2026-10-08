import copy
import json
import pytest
from unittest.mock import patch

from env_factory.sandbox_runtime import ContractModelMetricEvaluator, EpisodeStore


@pytest.mark.parametrize('phase', ['before', 'after_commit'])
@pytest.mark.parametrize('retry_key', ['same-key', 'different-key', None])
def test_actual_retry_request_identity_reaches_judge(tmp_path, phase, retry_key):
    from env_factory.sandbox_runtime import (BusinessLifecycle, ContractToolRegistry,
        ManifestDataStore, SandboxApplication)
    (tmp_path / 'schema.json').write_text(json.dumps({'columns': [{'name': 'id', 'type': 'INTEGER'}]}))
    (tmp_path / 'rows.jsonl').write_text('{"id": 1}\n')
    store = EpisodeStore(tmp_path / 'episode.sqlite3')
    data = ManifestDataStore({'environment_mode': 'stateful', 'tables': [{'table_name': 'items',
        'schema_file': 'schema.json', 'rows_file': 'rows.jsonl'}]}, tmp_path, store)
    data.reset()
    lifecycle = BusinessLifecycle({'version': '1.0', 'roles': [], 'tools': {}, 'dynamics': {
        'faults': [{'id': 'private-fault', 'tool': 'read', 'call_index': 1, 'phase': phase, 'status': 503},
                   {'id': 'future-fault', 'tool': 'read', 'call_index': 99, 'phase': 'before', 'status': 504}]}},
        data, store, ['read'])
    registry = ContractToolRegistry([{'type': 'function', 'function': {'name': 'read', 'parameters': {
        'type': 'object', 'properties': {'scope': {'type': 'string'}}, 'required': ['scope']}}}],
        {'read': lambda args: {'records': data.table('items')}}, event_recorder=store.event, lifecycle=lifecycle)
    app = SandboxApplication(episode_store=store, tool_registry=registry,
        observation=lambda: {}, reward=lambda: {}, user_turn=lambda messages: {})
    assert app.handle('POST', '/v1/tools/read', {'scope': 'A'}, {'Idempotency-Key': 'same-key'})[0] == 503
    headers = {'Idempotency-Key': retry_key} if retry_key else {}
    assert app.handle('POST', '/v1/tools/read', {'scope': 'A'}, headers)[0] == 200
    contract = {'metrics': [{'id': 'recovery', 'evaluation_inputs': ['final_agent_response', 'tool_results'],
        'evaluator': {'kind': 'external_llm_judge', 'score_mapping': {'pass': 1, 'fail': 0}}}]}
    observed = []
    def judge(client, messages, **kwargs):
        observed.append(json.loads(messages[1]['content'])['runtime_evidence'])
        return {'label': 'pass'}
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', new=judge):
        ContractModelMetricEvaluator(contract, store).evaluate_all(
            {'trajectory': store.replay(), 'final_agent_response': 'Retried with the same key.'}, {})
    attempts = observed[0]['observed_tool_attempts']
    assert [a['status'] for a in attempts] == [503, 200]
    assert all(a['tool_name'] == 'read' and a['arguments'] == {'scope': 'A'} for a in attempts)
    assert (attempts[0]['idempotency_key_sha256'] == attempts[1]['idempotency_key_sha256']) is (retry_key == 'same-key')
    assert attempts[1]['idempotency_key_sha256'] is None if retry_key is None else len(attempts[1]['idempotency_key_sha256']) == 64
    encoded = json.dumps(observed)
    for hidden in ['same-key', 'different-key', 'private-fault', 'future-fault']:
        assert hidden not in encoded


def test_judge_sees_observed_fault_then_retry_but_not_future_fault_schedule(tmp_path):
    store = EpisodeStore(tmp_path / 'episodes.sqlite3')
    store.reset(seed=17)
    contract = {'metrics': [{'id': 'recovery', 'category': 'outcome', 'scope': 'terminal',
        'rubric': 'Describe the actual transient failure and successful retry.',
        'evaluation_inputs': ['final_agent_response', 'tool_results'],
        'evaluator': {'kind': 'external_llm_judge', 'score_mapping': {'pass': 1, 'fail': 0}}}],
        'business_lifecycle': {'dynamics': {'faults': [{'tool': 'future_private_tool', 'call_index': 99}]}}}
    events = [
        {'event': 'injected_fault', 'payload': {'tool': 'read', 'status': 503, 'call_index': 1,
            'phase': 'before', 'id': 'private-fault-identifier'}, 'result': {'committed': False}},
        {'event': 'tool_call', 'payload': {'tool_name': 'read', 'arguments': {'scope': 'A'}},
            'result': {'records': [{'value': 7}]}},
    ]
    context = {'trajectory': {'events': events}, 'final_agent_response': 'The first query failed; retry succeeded.'}
    evidence = []
    def judge(client, messages, **kwargs):
        value = json.loads(messages[1]['content'])['runtime_evidence']
        evidence.append(value)
        return {'label': 'pass' if value.get('observed_tool_failures') else 'fail'}
    evaluator = ContractModelMetricEvaluator(contract, store)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', new=judge):
        assert evaluator.evaluate_all(context, {}) == {'recovery': 1}
        no_failure = copy.deepcopy(context)
        no_failure['trajectory']['events'] = events[1:]
        assert evaluator.evaluate_all(no_failure, {}) == {'recovery': 0}
    assert evidence[0]['observed_tool_failures'] == [{'tool_name': 'read', 'status': 503,
        'call_index': 1, 'phase': 'before', 'committed': False, 'trajectory_position': 0}]
    assert 'observed_tool_failures' not in evidence[1]
    assert 'future_private_tool' not in json.dumps(evidence)
    assert 'private-fault-identifier' not in json.dumps(evidence)
    assert len(evidence[0]['tool_results']) == 1
    assert len(evidence) == 2  # Changed failure history invalidates judge reuse.


def test_fault_evidence_respects_declared_tool_input_boundary(tmp_path):
    store = EpisodeStore(tmp_path / 'episodes.sqlite3')
    store.reset(seed=17)
    contract = {'metrics': [{'id': 'style', 'evaluation_inputs': ['final_agent_response'],
        'evaluator': {'kind': 'external_llm_judge', 'score_mapping': {'pass': 1, 'fail': 0}}}]}
    context = {'final_agent_response': 'An answer.', 'trajectory': {'events': [
        {'event': 'injected_fault', 'payload': {'tool': 'read', 'status': 503}, 'result': {'committed': False}}]}}
    captured = []
    def judge(client, messages, **kwargs):
        captured.append(json.loads(messages[1]['content'])['runtime_evidence'])
        return {'label': 'pass'}
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', new=judge):
        ContractModelMetricEvaluator(contract, store).evaluate_all(context, {})
    assert 'observed_tool_failures' not in captured[0]
