import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from env_factory_v2.examples import direct, answer, lit, scenario
from env_factory_v2.contracts.contract import compile_contract
from v2_business_acceptance import accept, business_truth, state_counterexamples


def test_wrong_state_is_rejected_without_answer_obligations():
    import copy
    from env_factory_v2.examples import transaction
    from measure_v2_natural import execute_reference
    compiled = compile_contract(transaction())
    scenario = compiled['source']['scenarios'][0]
    world = execute_reference(compiled, scenario)
    truth = {'answer': {}, 'initial': {'accounts': copy.deepcopy(compiled['source']['tables'][0]['rows'])},
             'final_state': copy.deepcopy(world.data)}
    checks = state_counterexamples(compiled, scenario, truth, Mock())
    assert len(checks) == 2
    assert all(check['passed'] for check in checks)


def test_empty_answer_mapping_uses_supported_native_schema():
    from env_factory_v2.contracts.contract import validate_schema, validate_value
    from v2_natural_semantics import schema
    output_schema = schema({}, ['committed'])
    validate_schema(output_schema)
    validate_value({'business_consistent': True, 'reason': 'state-only task', 'mapping': []}, output_schema)


def test_mapping_recovery_runs_full_acceptance_without_revising_business_verdict():
    source = direct()
    source['answers'] = [answer('result', '实付金额', lit(186.6), {'type': 'number'})]
    source['scenarios'] = [scenario('correct', [], {'result': lit(186.6)}),
                           scenario('wrong', [], {'result': lit(1)}, False)]
    entries = [{'role': 'payable', 'selectors': [{'field': 'result', 'path': ''}],
                'packing': 'single', 'time_unit': 'native', 'names': 'native'}]
    spec = {'kind': 'discount', 'params': {'prices': [128, 96], 'threshold': 200,
            'rate': 0.9, 'coupon': 15}, 'observations': {}, 'mutation': None,
            'answer_obligations': [{'role': 'payable', 'source': 'user_request', 'text': source['task']}]}
    for business_consistent in (True, False):
        agent = Mock()
        agent.ask.side_effect = [({'business_consistent': business_consistent,
            'reason': 'independent review', 'mapping': []}, {}), ({'mapping': entries}, {})]
        result = accept(compile_contract(source), {'raw_request': source['task']},
                        {'policy': '优惠规则', 'tables': []}, {'facts': {}}, spec, agent)
        assert result['passed'] is business_consistent
        assert result['review']['business_consistent'] is business_consistent
        assert result['review']['mapping_recovery']['original_mapping'] == []
        assert agent.ask.call_count == 2
        assert all(check['passed'] for check in result['checks'])


def test_failed_positive_reference_retains_business_evidence():
    source = direct()
    source['answers'] = [answer('result', '实付金额', lit(186.6), {'type': 'number'})]
    source['scenarios'] = [scenario('incorrect_reference', [], {'result': lit(1)}),
                           scenario('negative', [], {'result': lit(2)}, False)]
    entries = [{'role': 'payable', 'selectors': [{'field': 'result', 'path': ''}],
                'packing': 'single', 'time_unit': 'native', 'names': 'native'}]
    agent = Mock()
    agent.ask.return_value = ({'business_consistent': True, 'reason': 'reviewed', 'mapping': entries}, {})
    spec = {'kind': 'discount', 'params': {'prices': [128, 96], 'threshold': 200,
            'rate': 0.9, 'coupon': 15}, 'observations': {}, 'mutation': None,
            'answer_obligations': [{'role': 'payable', 'source': 'user_request', 'text': source['task']}]}
    result = accept(compile_contract(source), {'raw_request': source['task']},
                    {'policy': '优惠规则', 'tables': []}, {'facts': {}}, spec, agent)
    assert result['passed'] is False
    assert result['error'] == 'no independently correct positive reference'
    assert result['reference_attempts'][0]['scenario'] == 'incorrect_reference'
    assert any(c['id'] == 'answer_payable' and not c['passed'] for c in result['checks'])


def test_predeclared_answer_scope_keeps_complete_transaction_state():
    import copy
    import pytest
    from env_factory_v2.contracts.contract import ContractError
    from prepare_v2_natural_pilot import cases
    case = cases()[16]
    spec = copy.deepcopy(case['acceptance'])
    before = business_truth(spec, case['business'], case['user'], user_request='调拨12件')
    spec['answer_obligations'] = []
    after = business_truth(spec, case['business'], case['user'], user_request='调拨12件')
    assert after['answer'] == {}
    assert after['state_result_values'] == before['answer']
    assert after['final_state'] == before['final_state']
    assert after['initial'] == before['initial']
    spec['answer_obligations'] = [{'role': 'source_stock', 'source': 'user_request', 'text': '剩余库存'}]
    with pytest.raises(ContractError, match='public'):
        business_truth(spec, case['business'], case['user'], user_request='调拨12件')


def test_predeclared_projection_accepts_nested_answers_and_tests_wrong_values():
    source = direct()
    source['answers'] = [answer('result', '实付金额与节省金额', lit({'payable':186.6,'savings':37.4}),
        {'type':'object','properties':{'payable':{'type':'number'},'savings':{'type':'number'}},
         'required':['payable','savings'],'additionalProperties':False})]
    source['scenarios'] = [scenario('correct', [], {'result':lit({'payable':186.6,'savings':37.4})}),
        scenario('wrong', [], {'result':lit({'payable':1,'savings':1})},False)]
    entries = [{'role':r,'selectors':[{'field':'result','path':r}], 'packing':'single','time_unit':'native','names':'native'}
               for r in ['payable','savings']]
    agent = Mock()
    agent.ask.return_value = ({'business_consistent':True,'reason':'matches','mapping':entries},{'backend':'unit-test-only'})
    result = accept(compile_contract(source), {'raw_request':source['task']}, {'policy':'会员优惠规则','tables':[]}, {'facts':{}},
        {'kind':'discount','params':{'prices':[128,96],'threshold':200,'rate':0.9,'coupon':15},'observations':{},'mutation':None}, agent)
    assert result['passed']
    assert {c['id'] for c in result['checks']} >= {'wrong_payable','wrong_savings','caller_id_invariance'}
