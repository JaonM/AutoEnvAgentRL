"""Domain-neutral DAG contract tests on a small library, including hostile mutations."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from env_factory import sop
from env_factory.runtime import BusinessError, resolve_goal_params, write
from env_factory.production.world import goal_problems


@write
def lend(db, now, book: str):
    if not db['stock'][book]:
        db['loans'].append('bad partial mutation')
        raise BusinessError('unavailable')
    db['stock'][book] -= 1
    receipt = 'loan-' + str(len(db['loans']) + 7)
    db['loans'].append(receipt)
    return {'id': receipt}


@write
def return_book(db, now, receipt: str, book: str):
    if receipt not in db['loans']:
        raise BusinessError('unknown loan')
    db['loans'].remove(receipt); db['stock'][book] += 1
    return {'returned': True}


def read(db, now):
    return db['stock']


def atom(tool, reads, writes, properties=None):
    return dict(tool=tool, reads=reads, writes=writes,
                inputs={'type':'object','properties':properties or {},'required':list(properties or {}),'additionalProperties':False},
                output={'type':'object'})


def fixture():
    env=SimpleNamespace(make_tools=lambda:{'lend':lend,'return':return_book,'read':read})
    blueprint={'atoms':{'lend':atom('lend',['stock'],['stock','loans'],{'book':{'type':'string'}}),
        'return':atom('return',['loans'],['stock','loans'],{'receipt':{'type':'string'},'book':{'type':'string'}}),
        'read':atom('read',['stock'],[])}}
    check=lambda path,value:{'name':path,'assert':{'op':'eq','args':[{'path':path},value]}}
    case={'data':{'stock':{'b':1},'loans':[]},'graph':{'nodes':[
        {'id':'lend','atom':'lend','args':{'book':'b'},'checks':[check('after.stock.b',0)]},
        {'id':'return','atom':'return','args':{'book':'b','receipt':{'path':'outputs.lend.id'}},
         'checks':[check('after.stock.b',1),check('after.loans',[])]}],
        'edges':[{'from':'lend','to':'return','kind':'data'}]}}
    return env,blueprint,case


def test_runtime_receipt_binding_and_round_trip():
    env,b,c=fixture()
    assert sop.run_graph(env,b,c)['passed']
    assert sop.run_graph(env,b,c,reverse=True)['passed']


def test_mutation_outside_declared_frame_is_caught():
    env,b,c=fixture();b['atoms']['lend']['writes']=['stock']
    report=sop.run_graph(env,b,c)
    assert not report['passed'] and any('undeclared write' in p for p in report['problems'])


@pytest.mark.parametrize('fault',['cycle','missing_binding_edge','unknown_atom','conflict','wrong_type'])
def test_invalid_graphs_fail_closed(fault):
    env,b,c=fixture()
    if fault=='cycle':c['graph']['edges'].append({'from':'return','to':'lend','kind':'state'})
    if fault=='missing_binding_edge':c['graph']['edges'][0]['kind']='state'
    if fault=='unknown_atom':c['graph']['nodes'][0]['atom']='invented'
    if fault=='conflict':
        c['graph']['nodes'][1]['args']['receipt']='loan-7';c['graph']['edges']=[]
    if fault=='wrong_type':c['graph']['nodes'][0]['args']['book']=12
    assert not sop.run_graph(env,b,c)['passed']


def test_expected_failure_is_atomic_and_branch_runs():
    env,b,c=fixture();c['data']['stock']['b']=0
    first=c['graph']['nodes'][0];first['error']='unavailable';first['checks']=[]
    second=c['graph']['nodes'][1]
    second.update(atom='read',args={},when={'op':'eq','args':[{'path':'outputs.lend.error'},'unavailable']},
                  checks=[{'name':'rollback','assert':{'op':'eq','args':[{'path':'after.loans'},[]]}}])
    assert sop.run_graph(env,b,c)['passed']


def test_independent_reads_accept_reverse_order():
    env,b,c=fixture()
    c['graph']={'nodes':[dict(id=x,atom='read',args={},checks=[{'name':'read','assert':True}]) for x in ('a','b')],'edges':[]}
    a=sop.run_graph(env,b,c);z=sop.run_graph(env,b,c,reverse=True)
    assert a['passed'] and z['passed'] and a['outputs_sha256']==z['outputs_sha256']


def test_row_scope_protects_other_customer():
    a={'writes':['accounts.*.balance'],'row_scopes':[{'table':'accounts','key':'id','argument':'owner'}]}
    before={'accounts':[{'id':'a','balance':1},{'id':'b','balance':2}]};after=deepcopy(before)
    after['accounts'][1]['balance']=100
    assert sop.frame_problems(before,after,a,{'owner':'a'})==['write escaped object scope: accounts']


def test_goal_binding_validates_source_schema_and_never_guesses_id():
    schema={'lend':{'params':{},'outcome':{'type':'object','properties':{'id':{'type':'string'}}}},
            'inspect':{'params':{'receipt':'string'},'outcome':{'type':'object','properties':{}}}}
    goals=[{'process':'lend','params':{}},
           {'process':'inspect','params':{'receipt':None},'bindings':{'receipt':{'goal':1,'path':'id'}}}]
    assert goal_problems(goals,schema)==[]
    assert resolve_goal_params(goals[1],{'goal_1':{'id':'runtime-839'}})=={'receipt':'runtime-839'}
    goals[1]['params']['receipt']='guessed-001'
    assert goal_problems(goals,schema)
    with pytest.raises(ValueError):resolve_goal_params(goals[1],{'goal_1':{'id':'runtime-839'}})
    goals[1]['params']['receipt']=None;goals[1]['bindings']['receipt']['goal']=2
    assert goal_problems(goals,schema)


def test_user_simulator_cannot_repeat_planner_binding_metadata():
    from env_factory.serving.user_sim import customer_facts
    facts={'goals':[{'process':'inspect','params':{'id':None},'bindings':{'id':{'goal':1,'path':'secret_id'}}}]}
    shown=customer_facts(facts)
    assert 'bindings' not in shown['goals'][0]
    assert 'secret_id' not in str(shown)
    assert facts['goals'][0]['params']['id'] is None


def test_sop_is_not_an_agent_trace_rubric_and_arbitrary_composition_rejected():
    request={'sop_blueprint':{'methods':{}},'user_facts':{'goals':[{'process':'x'},{'process':'y'}]}}
    assert sop.task_problems(request)
    request['sop_legacy']=True
    assert sop.task_problems(request)==[]


def test_fake_causality_from_unrelated_persistent_edit_is_rejected():
    def expected(data,facts,now):
        state=deepcopy(data);answer={}
        for i,g in enumerate(facts['goals'],1):
            state[g['params']['field']]+=1
            answer[f'goal_{i}']={'status':'completed'}
        return {'answer':answer,'final_state':state}
    method={'processes':['edit','edit'],'edges':[{'from':0,'to':1,'kind':'state','causal':True}]}
    r={'sop_blueprint':{'methods':{'m':method}},'sop_method':'m','time':{'start':0},'tables':{'a':0,'b':0},
       'user_facts':{'goals':[{'process':'edit','params':{'field':x}} for x in ('a','b')]}}
    assert not sop.task_report(None,SimpleNamespace(expected=expected),r)['passed']
