import pytest

from env_factory.coverage import report, require


def test_observable_evidence_and_distinct_tasks():
    cells=[{'id':'refund','minimum':2,'predicates':[{'path':'observed.amount','max':-1},
             {'path':'tools','contains':'refund'}]}]
    good=dict(id='a',observed={'amount':-20},tools=['refund'],reference_passed=True,natural_passed=True)
    claimed=dict(id='b',label='refund',reference_passed=True,natural_passed=True)
    assert report(cells,[good,good,claimed])['reference_gaps']==['refund']
    second={**good,'id':'b','natural_passed':False}
    result=report(cells,[good,second])
    assert result['reference_covered']==1 and result['natural_covered']==0
    assert report(cells,[good,{**second,'natural_passed':True}])['natural_covered']==1


def test_nested_evidence_is_domain_independent():
    cell={'id':'lab','predicates':[{'path':'observations.*.result','equals':'negative'}]}
    row={'id':'1','observations':[{'result':'positive'},{'result':'negative'}],
         'reference_passed':False,'natural_passed':True}
    assert report([cell],[row])['natural_covered']==0
    assert report([cell],[{**row,'reference_passed':True}])['natural_covered']==1


@pytest.mark.parametrize('cells',[
    [{'id':'a','predicates':[]}],
    [{'id':'a','predicates':[{'path':'x'}],'minimum':0}],
    [{'id':'a','predicates':[{'path':'x'}]}]*2,
])
def test_invalid_contracts_rejected(cells):
    with pytest.raises(ValueError):report(cells,[])


def test_invalid_predicate_rejected_before_data_arrives():
    with pytest.raises(ValueError):
        report([{'id':'x','predicates':[{'path':'x','invented_operator':1}]}],[])


def test_natural_release_requires_natural_evidence():
    cells=[{'id':'x','predicates':[{'path':'outcome','equals':'done'}]}]
    row={'id':'task','outcome':'done','reference_passed':True}
    assert require(cells,[row],stage='reference')['reference_covered']==1
    with pytest.raises(ValueError):require(cells,[row])
    assert require(cells,[{**row,'natural_passed':True}])['natural_covered']==1
