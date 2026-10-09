import copy
import json
from unittest.mock import Mock, patch

import pytest

from env_factory_v2.challenge import challenge
from env_factory_v2.code_agent import NativeCodeAgent
from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import direct, simple
from env_factory_v2.pipeline import build


def native(tmp_path):
    return NativeCodeAgent(base_url='http://127.0.0.1:9999/v1', token='test-only-token-000000000000000000', evidence_root=tmp_path)


def test_connection_budget_reaches_container_environment(tmp_path, monkeypatch):
    from env_factory_v2.__main__ import environment_agent
    token = tmp_path / 'token'
    token.write_text('test-only-token-000000000000000000')
    connection = tmp_path / 'connection.json'
    connection.write_text(json.dumps({'backend': 'codex', 'model': 'gpt-6.1-sol',
        'base_url': 'http://127.0.0.1:9999/v1', 'token_file': str(token), 'timeout_seconds': 240}))
    agent = NativeCodeAgent.from_connection(connection, evidence_root=tmp_path)
    assert agent.timeout == 255
    monkeypatch.setenv('V2_BRIDGE_URL', agent.base_url)
    monkeypatch.setenv('V2_BRIDGE_TOKEN_FILE', str(token))
    monkeypatch.setenv('V2_BRIDGE_TRANSPORT_TIMEOUT', str(agent.timeout))
    assert environment_agent().timeout == 255


@pytest.mark.parametrize('timeout', [True, 0, -1, float('inf'), float('nan'), 616])
def test_native_transport_rejects_unbounded_or_invalid_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match='timeout'):
        NativeCodeAgent(base_url='http://127.0.0.1:9999/v1',
            token='test-only-token-000000000000000000', timeout=timeout, evidence_root=tmp_path)


def payload():
    return {'contract_identity': 'test-contract', 'field': {'criteria': [{'id': 'facts'}]},
            'authoritative_facts': {'ids': ['I1']}}


def judgment(passed=True):
    return ({'criteria': {'facts': passed}}, {'model': 'gpt-6.1-sol', 'backend': 'codex_cli', 'response_id': 'unit-test-only'})


def test_exact_judgment_reuse_and_fresh_independent_replay(tmp_path):
    agent = native(tmp_path)
    agent.ask = Mock(side_effect=[judgment(), judgment(), judgment()])
    first = agent.judge(payload())
    hit = agent.judge(payload())
    fresh = agent.independent_judge(payload())
    changed = payload()
    changed['authoritative_facts']['ids'] = []
    agent.judge(changed)
    assert agent.ask.call_count == 3
    assert hit['provenance']['reused'] is True
    assert first['provenance']['dependency_identity'] == fresh['provenance']['dependency_identity']
    assert not fresh['provenance']['reused']


def test_conflicting_fresh_judgment_poisoned_for_entire_scope(tmp_path):
    agent = native(tmp_path)
    agent.ask = Mock(side_effect=[judgment(True), judgment(False)])
    agent.judge(payload())
    with pytest.raises(RuntimeError, match='conflicts'):
        agent.independent_judge(payload())
    with pytest.raises(RuntimeError, match='poisoned'):
        agent.judge(payload())
    assert agent.ask.call_count == 2
    agent.start_scope()
    agent.ask = Mock(return_value=judgment(True))
    agent.judge(payload())
    assert agent.ask.call_count == 1


def test_native_user_expression_cannot_generate_new_facts_or_authorization(tmp_path):
    agent = native(tmp_path)
    agent.ask = Mock(return_value=({'language': 'zh', 'tone': 'polite'}, {'unit_test': True}))
    event = {'approved': False, 'proposal': {'tool': 'transfer', 'arguments': {'amount': 20}}}
    response = agent.express_user(event)
    assert response['event'] == event
    assert '不批准' in response['text'] and '20' in response['text']
    schema = agent.ask.call_args.args[2]
    assert set(schema['properties']) == {'language', 'tone'}


def test_unrelated_natural_question_cannot_unlock_a_user_stage(tmp_path):
    agent = native(tmp_path)
    agent.ask = Mock(return_value=({'stage_id': '__unrelated__'}, {'unit_test': True}))
    with pytest.raises(ValueError, match='does not request'):
        agent.interpret_user('天气怎么样？', simple()['user']['stages'])


def challenge_value(source):
    rules = [r['id'] for r in source['rules']]
    return {'business_consistent': True, 'reason': 'Unit-test transport fixture, not native proof.', 'cases': [
        {'id': 'independent_positive', 'rule_ids': rules, 'scenario_id': 'correct', 'reason': 'correct answer',
         'mutations_json': '[]', 'answer_json': '{}', 'expect_success': True},
        {'id': 'independent_negative', 'rule_ids': rules, 'scenario_id': 'wrong', 'reason': 'wrong answer',
         'mutations_json': '[]', 'answer_json': '{}', 'expect_success': False}]}


def test_unexecuted_rejection_is_not_grounds_for_author_repair(tmp_path):
    source = direct()
    value = challenge_value(source)
    value['business_consistent'] = False
    agent = native(tmp_path)
    agent.ask = Mock(return_value=(value, {'unit_test': True}))
    with pytest.raises(RuntimeError, match='inconclusive'):
        challenge(compile_contract(source), {'goal': source['task']}, agent)


def test_contradictory_positive_label_is_audited_without_dropping_counterexample(tmp_path):
    source = direct()
    value = challenge_value(source)
    mislabeled = copy.deepcopy(value['cases'][1])
    mislabeled.update(id='incorrect_positive', expect_success=True, reason='This answer is deliberately wrong.')
    value['cases'].append(mislabeled)
    audit = {'business_consistent': True, 'reason': 'The contract follows the original fixed fact.',
             'cases': [{'id': c['id'], 'expect_success': c['id'] == 'independent_positive',
                        'reason': 'Original temperature is 18; 19 is incorrect.'} for c in value['cases']]}
    agent = native(tmp_path)
    agent.ask = Mock(side_effect=[(value, {'unit_test': 'review'}), (audit, {'unit_test': 'audit'})])
    result = challenge(compile_contract(source), {'goal': source['task']}, agent)
    assert result['passed'] and len(result['records']) == 3
    assert not result['records'][-1]['case']['expect_success']
    assert not result['review_audit']['original_review']['records'][-1]['passed']
    assert agent.ask.call_count == 2


def test_audit_cannot_make_real_correct_behavior_penalty_pass(tmp_path):
    source = direct()
    source['answers'][0]['expected'] = {'literal': 99}
    value = challenge_value(source)
    audit = {'business_consistent': False, 'reason': 'Original temperature is 18, not 99.',
             'cases': [{'id': c['id'], 'expect_success': c['expect_success'],
                        'reason': 'Use original business value, not grader target.'} for c in value['cases']]}
    agent = native(tmp_path)
    agent.ask = Mock(side_effect=[(value, {'unit_test': 'review'}), (audit, {'unit_test': 'audit'})])
    result = challenge(compile_contract(source), {'goal': 'Return the public temperature 18.'}, agent)
    assert not result['passed']
    assert not result['records'][0]['passed']
    assert result['records'][0]['case']['expect_success']


def test_audit_cannot_drop_an_inconvenient_case(tmp_path):
    source = direct()
    value = challenge_value(source)
    value['business_consistent'] = False
    audit = {'business_consistent': True, 'reason': 'Unsupported shortcut.', 'cases': [
        {'id': 'independent_positive', 'expect_success': True, 'reason': 'Keep only positive.'}]}
    agent = native(tmp_path)
    agent.ask = Mock(side_effect=[(value, {'unit_test': 'review'}), (audit, {'unit_test': 'audit'})])
    with pytest.raises(RuntimeError, match='preserve every independent case'):
        challenge(compile_contract(source), {'goal': source['task']}, agent)


def test_review_cannot_invent_invalid_action_error_expectations(tmp_path):
    source = direct()
    value = challenge_value(source)
    value['cases'][1]['expect_error'] = 'invented binding message'
    response = {'model': 'gpt-6.1-sol', 'backend': 'codex_cli', 'id': 'unit-test-only',
                'choices': [{'message': {'content': json.dumps(value)}}]}
    agent = native(tmp_path / 'native')
    agent.generate = Mock()
    with patch('env_factory_v2.code_agent.urlopen') as transport:
        transport.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
        result = build({'goal': source['task']}, agent=agent, source=source,
                       output=tmp_path / 'run', repairs=2, docker=False)
    assert result['status'] == 'infrastructure_failed' and not result['qualified']
    agent.generate.assert_not_called()


def test_invalid_independent_mutation_is_not_an_author_defect(tmp_path):
    source = simple()
    value = challenge_value(source)
    value['cases'][0]['mutations_json'] = json.dumps([{'table': 'items', 'key': 'missing', 'column': 'size', 'value': 'L'}])
    agent = native(tmp_path)
    agent.ask = Mock(return_value=(value, {'unit_test': True}))
    with pytest.raises(RuntimeError, match='ungrounded claim'):
        challenge(compile_contract(source), {'goal': source['task']}, agent)


def test_no_container_run_cannot_claim_qualified(tmp_path):
    source = direct()
    agent = native(tmp_path / 'native')
    agent.ask = Mock(return_value=(challenge_value(source), {'unit_test': True}))
    result = build({'category': source['category'], 'goal': source['task']}, agent=agent,
                   output=tmp_path / 'run', source=source, docker=False, repairs=0)
    assert result['qualified'] is False
    assert result['gates']['joint_execution_and_counterexamples'] is True
    assert result['gates']['container_execution'] is False
    assert (tmp_path / 'run/sandbox/Dockerfile').is_file()


def test_native_transport_failure_never_triggers_author_repair(tmp_path):
    agent = native(tmp_path / 'native')
    agent.generate = Mock(side_effect=RuntimeError('bridge unavailable'))
    result = build({'category': 'direct_response', 'goal': 'test'}, agent=agent, output=tmp_path / 'run', repairs=2)
    assert agent.generate.call_count == 1
    assert result['qualified'] is False and result['status'] == 'infrastructure_failed'


def test_broken_negative_recipe_is_rejected_before_any_review_model_call(tmp_path):
    source = direct()
    source['scenarios'][1]['answer']['temperature'] = {'ref': 'evidence.never_executed.value'}
    agent = native(tmp_path / 'native')
    agent.ask = Mock()
    result = build({'goal': source['task']}, agent=agent, output=tmp_path / 'run', source=source, repairs=0, docker=False)
    assert not result['qualified']
    assert result['attempts'][0]['failed_phase'] == 'recipe_preflight'
    agent.ask.assert_not_called()


def test_platform_change_during_native_review_invalidates_evidence(tmp_path):
    source = direct()
    agent = native(tmp_path / 'native')
    agent.ask = Mock(return_value=(challenge_value(source), {'unit_test': True}))
    with patch('env_factory_v2.pipeline.implementation_identity', side_effect=['before', 'before', 'after']):
        result = build({'goal': source['task']}, agent=agent, output=tmp_path / 'run', source=source, repairs=0, docker=False)
    assert result['qualified'] is False and result['status'] == 'infrastructure_failed'
    assert not (tmp_path / 'run/sandbox').exists()


def test_audit_appends_real_negative_without_relabeling_correct_cases(tmp_path):
    source = direct()
    value = challenge_value(source)
    value['cases'][1].update(scenario_id='correct', expect_success=False)
    audit = {'business_consistent': True, 'reason': 'Both prior cases return 18.', 'cases': [
        {'id': case['id'], 'expect_success': True, 'reason': '18 is the requested fact.'}
        for case in value['cases']]}
    extra = copy.deepcopy(value['cases'][1])
    extra.update(id='actual_wrong_fact', scenario_id='wrong', reason='19 contradicts original temperature.')
    agent = native(tmp_path)
    agent.ask = Mock(side_effect=[(value, {}), (audit, {}), ({'cases': [extra]}, {})])
    result = challenge(compile_contract(source), {'goal': source['task']}, agent)
    assert result['passed']
    assert [record['case']['expect_success'] for record in result['records']] == [True, True, False]
    assert result['records'][-1]['first']['outcome']['reward'] == 0
    assert len(result['review_audit']['original_review']['records']) == 2


def test_supplement_that_is_actually_correct_cannot_manufacture_negative_proof(tmp_path):
    source = direct()
    value = challenge_value(source)
    value['cases'][1].update(scenario_id='correct', expect_success=False)
    audit = {'business_consistent': True, 'reason': 'Both answers correct.', 'cases': [
        {'id': case['id'], 'expect_success': True, 'reason': 'Correct fixed fact.'} for case in value['cases']]}
    extra = copy.deepcopy(value['cases'][1])
    extra.update(id='fake_negative', reason='Incorrect negative label for correct answer.')
    agent = native(tmp_path)
    agent.ask = Mock(side_effect=[(value, {}), (audit, {}), ({'cases': [extra]}, {})])
    result = challenge(compile_contract(source), {'goal': source['task']}, agent)
    assert not result['passed']
    assert not result['records'][-1]['passed']
