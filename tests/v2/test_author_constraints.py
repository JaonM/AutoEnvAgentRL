import copy
import pytest

from env_factory_v2.author_workbench import AuthorWorkbench
from env_factory_v2.examples import simple


def inputs(source):
    return {'goal':source['task'],'category':source['category'],
            'private_user_fixture':{'facts':copy.deepcopy(source['user']['facts'])},
            'business_evidence':{'tables':[{'id':t['id'],'rows':copy.deepcopy(t['rows'])}
                                          for t in source['tables']]}}


@pytest.mark.parametrize('change',['task','category','facts','rows'])
def test_author_checks_frozen_inputs_before_recipe_execution(change):
    source=simple()
    request=inputs(source)
    if change=='task':source['task']='different user request'
    elif change=='category':source['category']='ordinary_agentic'
    elif change=='facts':source['user']['facts']['size']='L'
    else:source['tables'][0]['rows'][0]['display']='changed display'
    result=AuthorWorkbench(source=source,request=request).call('check_contract',{})
    assert not result['passed']


def test_draft_cannot_mutate_operator_request_and_row_reordering_is_valid():
    source=simple()
    request=inputs(source)
    bench=AuthorWorkbench(source=source,request=request)
    request['goal']='later outside change'
    assert bench.call('check_contract',{})['passed']
