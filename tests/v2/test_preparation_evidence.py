import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/experiments'))
from v2_preparation_evidence import PreparationEvidence,request_identity
from env_factory_v2.native.code_agent import MODEL,object_schema
from env_factory_v2.contracts.expressions import ContractError


def fixture(tmp_path):
    schema=object_schema({'passed':{'type':'boolean'}})
    record={'input':{'source':'original'},'response':{'passed':False},'provenance':{
        'model':MODEL,'backend':'codex_cli','response_id':'original-native',
        'input_identity':request_identity('review',{'source':'original'},schema)}}
    old=tmp_path/'old';old.mkdir();(old/'0001.json').write_text(json.dumps(record))
    agent=SimpleNamespace(provenance=[],evidence_root=tmp_path/'new',ask=Mock(return_value=({'passed':True},{})))
    return schema,record,old,agent


def test_failed_review_is_reused_without_resampling_until_pass(tmp_path):
    schema,_,old,agent=fixture(tmp_path)
    cached=PreparationEvidence(agent,old)
    value,provenance=cached.ask('review',{'source':'original'},schema)
    assert value=={'passed':False}
    assert provenance['reuse_scope']=='preparation_exact_request'
    agent.ask.assert_not_called()
    assert (agent.evidence_root/'0001.json').exists()


@pytest.mark.parametrize('change',['prompt','input','schema'])
def test_changed_dependency_requires_new_model_review(tmp_path,change):
    schema,_,old,agent=fixture(tmp_path)
    role,payload='review',{'source':'original'}
    if change=='prompt':role+=' revised'
    elif change=='input':payload['source']='new'
    else:schema=object_schema({'passed':{'type':'boolean'},'reason':{'type':'string'}})
    PreparationEvidence(agent,old).ask(role,payload,schema)
    agent.ask.assert_called_once()


def test_conflicting_previous_decisions_are_not_cherry_picked(tmp_path):
    _,record,old,agent=fixture(tmp_path)
    other=copy.deepcopy(record);other['response']['passed']=True
    (old/'0002.json').write_text(json.dumps(other))
    with pytest.raises(ContractError,match='conflicting'):
        PreparationEvidence(agent,old)
