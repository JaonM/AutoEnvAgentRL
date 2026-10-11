import copy
import json

from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import direct, simple, transaction
from env_factory_v2.requirements.obligation_index import obligation_index, reference_changes
from env_factory_v2.authoring.author_workbench import AuthorWorkbench
from env_factory_v2.validation.requirement_review import mandatory_obligations


def index(source):
    return obligation_index(compile_contract(source))


def test_index_reuses_review_catalog_and_does_not_modify_business_source():
    for factory in (direct, simple, transaction):
        source=factory();before=copy.deepcopy(source);compiled=compile_contract(source)
        value=obligation_index(compiled)
        assert [{k:v for k,v in x.items() if k!='identity'} for x in value['obligations']]==mandatory_obligations(compiled['source'])
        assert source==before and 'passed' not in value
        assert not reference_changes(value,copy.deepcopy(value))['references_requiring_validation']


def test_changed_obligation_invalidates_references_even_with_unchanged_answers():
    source=simple();before=index(source)
    source['answers'][0]['meaning'] += ' Explain the requested scope.'
    after=index(source);changes=reference_changes(before,after)
    assert '/answers/0' in changes['changed_obligation_paths']
    assert len(changes['references_requiring_validation'])==len(after['references'])


def test_recipe_change_invalidates_its_inherited_probes_not_unrelated_scenarios():
    source=simple();before=index(source)
    selected=source['scenarios'][0]
    selected['steps']=copy.deepcopy(selected['steps'])
    selected['steps'].append({'kind':'wait','seconds':1})
    changes=reference_changes(before,index(source))
    affected={(x['kind'],x['id']) for x in changes['references_requiring_validation']}
    assert ('scenario',selected['id']) in affected
    assert all(('scenario',s['id']) not in affected for s in source['scenarios'][1:])
    for probe in source['probes']:
        assert (('probe',probe['id']) in affected)==(probe['scenario']==selected['id'])
    assert not changes['changed_obligation_paths']


def test_environment_change_invalidates_even_when_obligation_text_is_unchanged():
    source=transaction();before=index(source)
    source['tables'][0]['rows'][0]['balance'] += 1
    after=index(source);changes=reference_changes(before,after)
    assert len(changes['references_requiring_validation'])==len(after['references'])


def test_workbench_reports_stale_references_without_changing_validation_verdict():
    source=direct();bench=AuthorWorkbench(source=source)
    assert bench.call('check_contract',{})['passed']
    bench.call('edit_contract',{'edits':[{'op':'replace','path':'/answers/0/meaning',
        'value_json':json.dumps('Report the requested temperature accurately.')} ]})
    result=bench.call('check_contract',{})
    assert result['passed'] and result['reference_changes']['references_requiring_validation']
    assert not bench.call('check_contract',{})['reference_changes']['references_requiring_validation']
