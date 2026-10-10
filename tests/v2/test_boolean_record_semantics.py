import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/experiments'))
from v2_natural_semantics import normalize, fill_projections, wrong_projection
from env_factory_v2.expressions import ContractError


ENTRY = {'role':'eligible','selectors':[{'field':'records','path':''}],
         'packing':'single','time_unit':'native','names':'native','representation':'records',
         'record_layout':{'identity_path':'id','details_path':'','scope_field':'',
                          'columns':[{'name':'eligible','path':'can_renew'}]}}
DESCRIPTOR = {'kind':'selected_boolean','identity':'loan_id','fields':['eligible'],
              'member':'L1','complete_records':[{'loan_id':'L1','eligible':True},
                                              {'loan_id':'L2','eligible':False}]}
ROLES = {'eligible':DESCRIPTOR}
ANSWER = {'records':[{'id':'L2','can_renew':False,'note':'keep'},
                     {'id':'L1','can_renew':True,'note':'keep'}]}


def project(answer):
    return normalize(answer,[ENTRY],role_types={'eligible':bool},record_roles=ROLES)


def test_complete_record_truth_is_checked_before_selected_boolean():
    assert project(ANSWER)=={'eligible':True}
    inverse=fill_projections(ANSWER,[ENTRY],{'eligible':True},record_roles=ROLES)
    assert inverse==ANSWER
    wrong=wrong_projection(ANSWER,ENTRY,record_roles=ROLES)
    assert wrong['records'][0]==ANSWER['records'][0]
    assert wrong['records'][1]['note']=='keep'
    with pytest.raises(ContractError,match='complete qualification records'):
        project(wrong)


@pytest.mark.parametrize('defect',['unrelated_wrong','omit','extra','duplicate','integer_boolean'])
def test_other_record_errors_cannot_be_hidden_by_correct_selected_value(defect):
    answer=copy.deepcopy(ANSWER)
    if defect=='unrelated_wrong':answer['records'][0]['can_renew']=True
    elif defect=='omit':answer['records'].pop(0)
    elif defect=='extra':answer['records'].append({'id':'foreign','can_renew':False})
    elif defect=='duplicate':answer['records'].append(copy.deepcopy(answer['records'][0]))
    else:answer['records'][0]['can_renew']=0
    with pytest.raises(ContractError):project(answer)
