import copy
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_natural_semantics import normalize, fill_projections, wrong_projection
from env_factory_v2.expressions import ContractError

D = {'identity': 'node_id', 'scope': 'write', 'fields': ['allowed', 'allow_sources', 'deny_sources', 'reason'],
     'aliases': {'reason': {'allowed': 'effective_allow'}}}
E = {'role': 'files', 'selectors': [{'field': 'report', 'path': ''}], 'packing': 'single',
     'time_unit': 'native', 'names': 'native', 'representation': 'records',
     'record_layout': {'identity_path': 'file_id', 'details_path': 'permissions', 'scope_field': 'permission',
       'columns': [{'name': k, 'path': v} for k,v in [('allowed','allowed'), ('allow_sources','allow_ids'),
                   ('deny_sources','deny_ids'), ('reason','reason')]]}}
A = {'report': [{'file_id':'F1', 'chain':['F1','N1'], 'permissions': [
    {'permission':'read','allowed':False,'allow_ids':[],'deny_ids':['D1'],'reason':'explicit_deny'},
    {'permission':'write','allowed':True,'allow_ids':['G1'],'deny_ids':[],'reason':'allowed'}]}]}
T = [{'node_id':'F1','allowed':True,'allow_sources':['G1'],'deny_sources':[],'reason':'effective_allow'}]

def project(answer, entry=E):
    return normalize(answer, [entry], role_types={'files':list}, record_roles={'files':D})['files']


def test_nested_aliases_preserve_object_and_permission_scope():
    assert project(A) == T
    changed = copy.deepcopy(A)
    changed['report'][0]['permissions'][1]['allowed'] = False
    assert project(changed) != T
    assert A['report'][0]['permissions'][1]['allowed'] is True


def test_inverse_updates_only_requested_permission_and_preserves_extra_details():
    desired = [{'node_id':'F1','allowed':False,'allow_sources':[], 'deny_sources':['D2'],'reason':'explicit_deny'}]
    out = fill_projections(A, [E], {'files': desired}, record_roles={'files':D})
    assert project(out) == desired
    assert out['report'][0]['permissions'][0] == A['report'][0]['permissions'][0]
    assert out['report'][0]['chain'] == ['F1','N1']


@pytest.mark.parametrize('mutation', ['duplicate_object','duplicate_permission','missing_permission'])
def test_ambiguous_or_missing_identity_and_scope_rejected(mutation):
    answer = copy.deepcopy(A)
    if mutation == 'duplicate_object': answer['report'].append(copy.deepcopy(answer['report'][0]))
    if mutation == 'duplicate_permission': answer['report'][0]['permissions'].append(copy.deepcopy(answer['report'][0]['permissions'][1]))
    if mutation == 'missing_permission': answer['report'][0]['permissions'].pop()
    with pytest.raises(ContractError): project(answer)


def test_omitted_objects_do_not_equal_truth_and_inverse_cannot_invent_objects():
    assert project({'report': []}) != T
    with pytest.raises(ContractError):
        fill_projections({'report':[]}, [E], {'files':T}, record_roles={'files':D})


def test_model_cannot_select_scope_or_supply_arbitrary_transformations():
    entry = copy.deepcopy(E)
    entry['record_layout']['columns'][0]['path'] = 'permissions[?write]'
    with pytest.raises(ContractError): project(A, entry)
    with pytest.raises(ContractError):
        normalize(A, [E], role_types={'files':list})


def test_flat_canonical_records_supported():
    entry = copy.deepcopy(E)
    entry['record_layout'] = {'identity_path':'node_id', 'details_path':'', 'scope_field':'',
        'columns':[{'name':k,'path':k} for k in D['fields']]}
    assert project({'report':T},entry) == T


def test_negative_projection_changes_business_value_inside_scoped_record():
    out = wrong_projection(A, E, record_roles={'files':D})
    assert project(out)[0]['allowed'] is False
    assert out['report'][0]['permissions'][0] == A['report'][0]['permissions'][0]
    assert out['report'][0]['file_id'] == 'F1'


@pytest.mark.parametrize('packing', ['single', 'list'])
def test_single_array_record_packing_round_trips_and_supports_counterexamples(packing):
    entry = copy.deepcopy(E)
    entry['packing'] = packing
    assert project(A,entry) == T
    assert fill_projections(A,[entry],{'files':T},record_roles={'files':D}) == A
    assert project(wrong_projection(A,entry,record_roles={'files':D}),entry)[0]['allowed'] is False
