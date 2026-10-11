import copy
from unittest.mock import Mock, patch

import pytest

from env_factory_v2.contract import ContractError
from env_factory_v2.examples import direct
from env_factory_v2.pipeline import build
from env_factory_v2.verification import qualify, attach_independent_review


def review(compiled, passed=True):
    return {'scope':'native_independent_contract_challenge',
            'contract_identity':compiled['identity'],'passed':passed}


def test_later_review_finalizes_existing_execution_without_judging_again():
    c,r=qualify(direct()); before=copy.deepcopy(r)
    assert not r['passed'] and r['gates']['joint_execution_and_counterexamples']
    result=attach_independent_review(c,r,review(c))
    assert result['passed'] and result['records']==r['records'] and r==before
    assert result['evidence_identity'] != r['evidence_identity']
    assert not attach_independent_review(c,r,review(c,False))['passed']


@pytest.mark.parametrize('change',['evidence','review_identity','execution_identity'])
def test_review_cannot_be_attached_to_changed_or_unrelated_execution(change):
    c,r=qualify(direct()); v=review(c)
    if change=='evidence':r['records'].clear()
    elif change=='review_identity':v['contract_identity']='other'
    else:c['identity']='other'
    with pytest.raises(ContractError):attach_independent_review(c,r,v)


def test_failed_execution_does_not_hide_independent_contract_findings(tmp_path):
    import hashlib
    import json
    from env_factory_v2.expressions import canonical
    source=direct();c,r=qualify(source)
    r['gates']['joint_execution_and_counterexamples']=False
    r.pop('evidence_identity')
    r['evidence_identity']=hashlib.sha256(canonical(r).encode()).hexdigest()
    agent=Mock()
    with patch('env_factory_v2.pipeline.qualify',return_value=(c,r)), \
         patch('env_factory_v2.pipeline.challenge',return_value=review(c,False)) as challenge:
        result=build({'category':source['category'],'goal':source['task']},agent=agent,
            output=tmp_path/'run',source=source,docker=False,repairs=0,tool_interference=0)
    assert not result['qualified'] and challenge.call_count==1
    feedback=json.loads((tmp_path/'run/attempt-1/feedback.json').read_text())
    assert not feedback['verification']['gates']['joint_execution_and_counterexamples']
    assert not feedback['independent_challenge']['passed']


def test_successful_execution_is_not_repeated_after_independent_review(tmp_path):
    source=direct();c,_=qualify(source);agent=Mock()
    with patch('env_factory_v2.pipeline.qualify',wraps=qualify) as execution, \
         patch('env_factory_v2.pipeline.challenge',return_value=review(c)) as challenge:
        result=build({'category':source['category'],'goal':source['task']},agent=agent,
            output=tmp_path/'run',source=source,docker=False,repairs=0,tool_interference=0)
    assert execution.call_count==1 and challenge.call_count==1
    assert result['gates']['independent_business_challenge']
    assert list(result['attempts'][0]['phases']).index('execution') < list(result['attempts'][0]['phases']).index('independent_challenge')
    assert not result['qualified']  # Container gate remains mandatory.
