import copy
import json
from unittest.mock import Mock,patch
import pytest
from test_independent_evidence_reuse import explained
from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.contracts.expressions import ContractError
from env_factory_v2.validation.challenge import _complete_case_answers, _case_scenario


def fixture():
    c=compile_contract(explained());s=c['source']['scenarios'][0]
    field=c['source']['explanations'][0]['id']
    answer=copy.deepcopy(s['answer']);del answer[field]
    case={'id':'case','scenario_id':s['id'],'answer_json':json.dumps(answer),
          'mutations_json':'[]','user_overrides_json':'{}','steps_json':'null','expect_success':True}
    return c,field,case


def test_omitted_field_completion_cannot_change_existing_case_semantics():
    c,field,case=fixture();agent=Mock()
    agent.ask.return_value=({'repairs':[{'index':0,'field':field,'expression_json':json.dumps({'literal':'Independent explanation'})}]},{})
    value,receipt=_complete_case_answers(c,{},agent,{'cases':[case]})
    for key in case:
        if key!='answer_json':assert value['cases'][0][key]==case[key]
    old=json.loads(case['answer_json']);new=json.loads(value['cases'][0]['answer_json'])
    assert all(new[k]==v for k,v in old.items()) and field in new
    assert receipt['original_cases']==[case]


def test_completion_cannot_overwrite_an_existing_answer():
    c,field,case=fixture();existing=next(iter(json.loads(case['answer_json'])))
    agent=Mock();agent.ask.return_value=({'repairs':[{'index':0,'field':existing,'expression_json':'{"literal":0}'}]}, {})
    with pytest.raises(ContractError):_complete_case_answers(c,{},agent,{'cases':[case]})


def test_inherited_trajectory_also_validates_complete_answer_schema():
    c,_,case=fixture();scenarios={s['id']:s for s in c['source']['scenarios']}
    with pytest.raises(ContractError,match='answer fields mismatch'):_case_scenario(c,case,scenarios)


def test_bad_case_labels_are_audited_even_with_a_separate_clause_defect():
    from test_requirement_review import fixture as reviewed_fixture
    from env_factory_v2.validation.challenge import challenge
    c,_,review=reviewed_fixture()
    review['business_consistent']=False
    review['requirement_reviews'][0]['consistent']=False
    bad={**copy.deepcopy(review['cases'][1]),'id':'wrong_positive','expect_success':True}
    review['cases'].append(bad)
    audit={'business_consistent':True,'reason':'Corrected labels do not resolve the separate clause finding.',
           'cases':[{'id':x['id'],'expect_success':x['id']=='yes','reason':'Original task truth.'} for x in review['cases']]}
    agent=Mock();agent.ask.side_effect=[(review,{}),(audit,{})]
    result=challenge(c,{},agent)
    assert agent.ask.call_count==2 and all(r['passed'] for r in result['records'])
    assert not result['passed'] and not result['requirement_review']['passed']
    assert not result['review_audit']['original_review']['records'][-1]['passed']
