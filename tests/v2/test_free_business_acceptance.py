import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from env_factory_v2.examples import direct, answer, lit, scenario
from env_factory_v2.contract import compile_contract
from v2_business_acceptance import accept


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
