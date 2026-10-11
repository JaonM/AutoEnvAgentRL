import copy
from unittest.mock import Mock, patch

from env_factory_v2.examples import direct, lit
from env_factory_v2.verification import qualify, run_scenario
from env_factory_v2.code_agent import NativeCodeAgent
from env_factory_v2.contract import compile_contract
from env_factory_v2.challenge import _execute_case


def explained():
    source = direct()
    source['answers'][0]['weight'] = 0.5
    source['explanations'] = [{'id': 'reason', 'meaning': '解释公开温度', 'weight': 0.5,
        'criteria': [{'id': 'fact', 'rule_id': 'fixed_fact', 'obligation': '符合公开温度事实'}],
        'context_fields': ['temperature']}]
    for case in source['scenarios']:
        case['answer']['reason'] = lit('公开温度是18摄氏度。')
    extra = copy.deepcopy(source['scenarios'][0])
    extra['id'] = 'paraphrase'
    extra['answer']['reason'] = lit('18摄氏度是给定温度。')
    source['scenarios'].append(extra)
    return source


def judgment(passed=True):
    return {'criteria': {'fact': passed}, 'provenance': {'response_id': 'unit-only'}}


def test_unique_evidence_review_preserves_every_execution_and_scope_isolation():
    source = explained()
    primary = Mock(side_effect=lambda payload: judgment())
    independent = Mock(side_effect=lambda payload: judgment())
    with patch('env_factory_v2.verification.run_scenario', wraps=run_scenario) as execute:
        _, report = qualify(source, judge=primary, independent_judge=independent)
    assert execute.call_count == 6
    assert primary.call_count == 3
    assert independent.call_count == 2  # same text and evidence, then paraphrase
    assert report['gates']['joint_execution_and_counterexamples']
    evidence = report['records'][1]['repeat']['outcome']['semantic_evidence']['reason']
    assert evidence['provenance']['reuse_scope'] == 'qualification_independent_evidence'
    qualify(source, judge=primary, independent_judge=independent)
    assert independent.call_count == 4  # no cross-qualification reuse


def test_conflicting_independent_review_cannot_pass_by_reuse():
    _, report = qualify(explained(), judge=lambda p: judgment(),
                        independent_judge=lambda p: judgment(False))
    assert not report['gates']['joint_execution_and_counterexamples']
    assert not report['gates']['deterministic_reset_and_replay']


def native_judge(tmp_path):
    agent = NativeCodeAgent(base_url='http://127.0.0.1:9999/v1',
        token='test-only-token-000000000000000000', evidence_root=tmp_path)
    agent.ask = Mock(return_value=({'criteria': {'fact': True}, 'reasons': {'fact': {'reason': 'Test evidence.', 'obligation_quote': '符合公开温度事实', 'evidence': [{'path': '/field/criteria/0/obligation', 'quote': '符合公开温度事实'}]}}}, {'response_id': 'test-only'}))
    agent.independent_judge = Mock(side_effect=AssertionError('no implicit resampling in deterministic replay'))
    return agent


def test_native_replay_executes_twice_without_resampling_identical_semantics(tmp_path):
    agent = native_judge(tmp_path)
    with patch('env_factory_v2.verification.run_scenario', wraps=run_scenario) as execute:
        _, report = qualify(explained(), judge=agent.judge)
    assert execute.call_count == 6
    assert agent.ask.call_count == 2  # original text and distinct paraphrase
    assert report['gates']['joint_execution_and_counterexamples']
    assert report['gates']['deterministic_reset_and_replay']
    assert not report['gates']['independent_business_challenge']  # not bypassed by cached grades
    assert report['semantic_replay'] == 'same_judge_exact_evidence_reuse_when_supported'
    agent.independent_judge.assert_not_called()


def test_independent_case_uses_one_semantic_assessment_but_two_executions(tmp_path):
    compiled = compile_contract(explained())
    scenario = compiled['source']['scenarios'][0]
    case = {'id': 'external', 'scenario_id': scenario['id'], 'rule_ids': ['fixed_fact'],
            'reason': 'Test case', 'mutations_json': '[]', 'answer_json': '{}',
            'user_overrides_json': '{}', 'steps_json': 'null', 'expect_success': True}
    agent = native_judge(tmp_path)
    with patch('env_factory_v2.challenge.run_scenario', wraps=run_scenario) as execute:
        record = _execute_case(compiled, case, {scenario['id']: scenario}, agent)
    assert execute.call_count == 2 and agent.ask.call_count == 1
    assert record['passed']
    assert record['repeat']['outcome']['semantic_evidence']['reason']['provenance']['reused']
    agent.independent_judge.assert_not_called()


def test_reused_semantic_assessment_does_not_hide_execution_difference(tmp_path):
    agent = native_judge(tmp_path)
    calls = 0
    def corrupt_repeat(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = run_scenario(*args, **kwargs)
        if calls == 2:
            result['final_state']['unexpected'] = [{'id': 'wrong'}]
        return result
    with patch('env_factory_v2.verification.run_scenario', side_effect=corrupt_repeat):
        _, report = qualify(explained(), judge=agent.judge)
    assert not report['gates']['deterministic_reset_and_replay']
    assert not report['records'][0]['passed']
