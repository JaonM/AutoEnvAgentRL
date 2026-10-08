import copy,importlib.util
from pathlib import Path
import pytest
from scripts.sandbox import validate_agentic_training_value as module


class App:
    def __init__(self,constant=False,invalid=False):
        self.constant=constant;self.invalid=invalid;self.rows=[];self.events=[];self.answer=''
    def business_snapshot(self):return {'items':copy.deepcopy(self.rows)}
    def mutate_business_state(self,mutation):
        if self.invalid:raise ValueError('invalid data')
        for row in self.rows:
            if all(row[k]==v for k,v in mutation['selector'].items()):row.update(mutation['changes'])
    def handle(self,method,path,body=None,headers=None):
        if path=='/v1/reset':
            self.rows=[{'id':1,'archive':'A','theme':'old','quote':'evidence'}];self.events=[];self.answer=''
            return 200,{},{}
        if path=='/v1/tools/read':
            self.events.append({'event':'tool_call','payload':{'tool_name':'read','arguments':dict(body),'business_data_reads':['items']}})
            rows=[] if self.constant else [copy.deepcopy(r) for r in self.rows if r['archive']==body['archive'] and r['theme']==body['theme']]
            self.result=rows
            return 200,{'records':rows,'count':len(rows)},{}
        if path=='/v1/replay':return 200,{'events':self.events},{}
        if path=='/v1/agent_response':self.answer=body['content'];return 200,{},{}
        if path=='/v1/reward':
            # Original empty answer must lose credit when matching rows appear.
            score=float(not self.result)
            return 200,{'reward':score,'components':{'records':score}},{}
        raise AssertionError(path)

scenario={'steps':[{'operation':'reset'},{'operation':'tool_call','tool_name':'read','arguments':{'archive':'B','theme':'chosen'}},{'operation':'agent_response','content':'[]'},{'operation':'reward'}]}
manifest={'tables':[{'table_name':'items','primary_key':['id']}]}
impl=[{'tool_name':'read','table':'items','operation':'select','filters':[{'column':'archive','argument':'archive','operator':'eq'},{'column':'theme','argument':'theme','operator':'eq'}]}]
metrics=[{'id':'records','category':'outcome','weight':1,'score_range':[0,1]}]

def test_zero_row_select_gets_executable_membership_and_reward_counterexample():
    app=App();r=module.probe_reference_data_causality(app,{},scenario,manifest,impl)
    assert r['proved'] and r['field']=='query_membership'
    mutation=r['_probe_mutation']
    assert mutation['selector']=={'id':1}
    assert mutation['changes']=={'archive':'B','theme':'chosen'}
    sensitivity=module.probe_reference_data_reward_candidates(app,{},scenario,[mutation],1,metrics)
    assert sensitivity['proved']
    assert sensitivity['baseline_rewards']==[1,1]
    assert sensitivity['changed_rewards']==[0,0]

@pytest.mark.parametrize('kwargs',[{'constant':True},{'invalid':True}])
def test_constant_or_invalid_mutations_cannot_prove_sensitivity(kwargs):
    assert not module.probe_reference_data_causality(App(**kwargs),{},scenario,manifest,impl)['proved']

def test_pk_rewrite_is_not_used_for_membership():
    protected={'tables':[{'table_name':'items','primary_key':['id','archive']}]}
    assert not module.probe_reference_data_causality(App(),{},scenario,protected,impl)['proved']
