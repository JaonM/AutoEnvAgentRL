import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_communication_evidence import before_choice_messages, validate_delivery, communication_checks
from env_factory_v2.contracts.expressions import ContractError

SITES = [{'id':'S1','name':'家附近'}, {'id':'S2','name':'公司附近'}]


def world(trace):
    return SimpleNamespace(trace=trace, user=SimpleNamespace(stages={'choose': {'kind':'choice'}}),
        tools={'change': {'kind':'transaction'}})


def choice(text):
    return {'kind':'user','stage':'choose','question':text}


def test_only_actual_assistant_messages_up_to_choice_count():
    trace = [{'kind':'tool','tool':'query','result':{'items':SITES}},choice('可选公司附近，请选择'),
             {'kind':'user_dialogue','message':'还可选家附近'}]
    assert before_choice_messages(world(trace)) == [{'trace_index':1,'text':'可选公司附近，请选择'}]
    assert before_choice_messages(world([{'kind':'tool','status':'ok','tool':'change'}, choice('可选S2')])) == []


def verdict(identity='S2', quote='可选公司附近', index=0, offered=True):
    return {'clearly_offered':offered,'offers':[{'site_id':identity,'quote':quote,'trace_index':index}]}


def test_public_name_offer_without_final_answer_duplication():
    messages = before_choice_messages(world([choice('可选公司附近，请选择')]))
    assert validate_delivery(messages, verdict(), SITES, ['S2'])
    assert not validate_delivery(messages, verdict(), SITES, ['S1','S2'])
    assert not validate_delivery(messages, verdict(offered=False), SITES, ['S2'])


@pytest.mark.parametrize('change', [{'site_id':'S1'}, {'quote':'可选S2'}, {'trace_index':7}])
def test_fabricated_location_quote_or_trace_rejected(change):
    v = verdict()
    v['offers'][0].update(change)
    with pytest.raises(ContractError):
        validate_delivery([{'trace_index':0,'text':'可选公司附近'}],v,SITES,['S2'])


def test_identifier_prefix_is_not_an_offer():
    with pytest.raises(ContractError):
        validate_delivery([{'trace_index':0,'text':'可选S20'}],verdict(quote='可选S20'),SITES,['S2'])


def test_model_sees_messages_and_directory_but_not_independent_expected_set():
    class Agent:
        def ask(self, prompt, payload, schema):
            assert 'expected' not in str(payload)
            assert 'assistant_messages' in payload
            return verdict(), {'id':'test'}
    truth={'initial':{'sites':SITES},'communication':{'sites':{
        'kind':'location_options_before_choice','expected':['S2']}}}
    checks=communication_checks(world([choice('可选公司附近')]),truth,Agent())
    assert checks[0]['passed']


def test_tool_menu_or_generic_all_options_claim_does_not_prove_delivery():
    assert not validate_delivery([{'trace_index':0,'text':'以上是全部可选地点'}],
        {'clearly_offered':False,'offers':[]},SITES,['S2'])


def test_identical_extraction_reused_but_changed_truth_is_rechecked():
    class Agent:
        scope='first'
        calls=0
        def ask(self,*args):
            self.calls+=1
            return verdict(), {'response_id':'actual'}
    agent=Agent()
    truth={'initial':{'sites':SITES},'communication':{'sites':{
        'kind':'location_options_before_choice','expected':['S2']}}}
    w=world([choice('可选公司附近')])
    assert communication_checks(w,truth,agent)[0]['passed']
    truth['communication']['sites']['expected']=['S1']
    check=communication_checks(w,truth,agent)[0]
    assert not check['passed']
    assert check['provenance']['reused_identical_communication_evidence']
    assert agent.calls==1
    w.trace[0]['question']='仅可选公司附近'
    communication_checks(w,truth,agent)
    assert agent.calls==2
    agent.scope='second'
    communication_checks(w,truth,agent)
    assert agent.calls==3
