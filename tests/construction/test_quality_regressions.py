import importlib.util
import sys
import copy
import pytest
from pathlib import Path
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
from env_factory import sandbox_runtime as runtime
gate=load('reference_data_gate',Path(__file__).resolve().parents[2]/'scripts/sandbox/validate_agentic_training_value.py')

def event(tool,arguments,result):
 return {'event':'tool_call','payload':{'tool_name':tool,'arguments':arguments},'result':result}

def test_later_capture_recovers_but_order_still_required():
 wrong=event('lookup',{}, {'id':'wrong'});right=event('lookup',{}, {'id':'right'})
 use=event('use',{'id':'right'}, {})
 expected={'tool_name':'use','arguments':{'id':{'$ref':'id'}},'captures':[{'name':'id','tool_name':'lookup','path':'$.id'}]}
 check=lambda calls: runtime.DeclarativeMetricEvaluator._compare(calls,'contains_tool_call',expected)
 assert check([wrong,right,use])
 assert not check([use,right])
 assert not check([wrong,use])
 assert not check([right,event('use',{'id':'wrong'}, {})])

def test_capture_fields_must_come_from_same_producer_event():
 expected={'tool_name':'use','arguments':{'id':{'$ref':'id'},'version':{'$ref':'version'}},'captures':[{'name':key,'tool_name':'lookup','path':'$.'+key} for key in ['id','version']]}
 calls=[event('lookup',{}, {'id':'A','version':1}),event('lookup',{}, {'id':'B','version':2}),event('use',{'id':'A','version':2},{})]
 assert not runtime.DeclarativeMetricEvaluator._compare(calls,'contains_tool_call',expected)
 calls[-1]=event('use',{'id':'B','version':2},{})
 assert runtime.DeclarativeMetricEvaluator._compare(calls,'contains_tool_call',expected)

def test_reference_external_changes_do_not_allow_agent_changes():
 before={'rows':[{'id':1,'value':10},{'id':2,'value':20}]}
 replay={'events':[{'event':'external_business_change','payload':{'table':'rows','selector':{'id':1},'values':{'value':12}},'result':{'updated_count':1}}]}
 after=copy.deepcopy(before);after['rows'][0]['value']=12
 assert gate.reference_data_immutability(before,after,replay)['unchanged']
 after['rows'][1]['value']=21
 assert not gate.reference_data_immutability(before,after,replay)['unchanged']
 after['rows'][1]['value']=20;after['rows'][0]['value']=13
 assert not gate.reference_data_immutability(before,after,replay)['unchanged']
 assert not gate.reference_data_immutability(before,after,None)['unchanged']
 assert gate.reference_data_immutability(before,before,{'events':[]})['unchanged']
 assert before['rows'][0]['value']==10


def test_full_credit_sum_is_exact_without_rounding_partial_rewards():
    metrics=[{'id':str(i),'category':'outcome','weight':1/6,'score_range':[0,1]} for i in range(6)]
    aggregator=runtime.ContractRewardAggregator(metrics)
    assert aggregator.aggregate({str(i):1 for i in range(6)})['reward']==1.0
    scores={str(i):1 for i in range(6)};scores['0']=0
    assert aggregator.aggregate(scores)['reward'] < .84
    metrics.append({'id':'penalty','category':'penalty','weight':1,'score_range':[-1,0]})
    scores={str(i):1 for i in range(6)};scores['penalty']=-.5
    assert runtime.ContractRewardAggregator(metrics).aggregate(scores)['reward'] == pytest.approx(.5)


def test_failure_history_retains_pre_timeout_validation(tmp_path):
    import json
    from env_factory.generate_task import _generation_failure_history
    (tmp_path/'code_agent_generation.json').write_text(json.dumps({'attempts':[
        {'index':1,'status':'validation_failed','error':'reference_data_immutability','issue_ids':['rule-1']},
        {'index':2,'status':'failed','error':'CODE_AGENT_TIMEOUT'},
        {'index':3,'status':'passed'}]}))
    history=_generation_failure_history(tmp_path)
    assert [item['error'] for item in history] == ['reference_data_immutability','CODE_AGENT_TIMEOUT']
    assert history[0]['issue_ids']==['rule-1']


def test_recovery_adjudication_rejects_unbounded_or_mutating_probes(tmp_path):
    import json
    adjudicator=load('recovery_adjudicator',Path(__file__).resolve().parents[2]/'scripts/sandbox/adjudicate_review.py')
    task={'environment_plan':{'mode':'reference_data'},'metrics':[{'type':'rule-based','category':'outcome'}],
          'tool_implementations':[{'tool_name':'read','operation':'select'},{'tool_name':'write','operation':'update'}],
          'acceptance_contract':{'executable_scenarios':[{'kind':'goal_success','scenario_id':'reference',
              'steps':[{'operation':'reset'},{'operation':'reward'}]}]}}
    (tmp_path/'task.json').write_text(json.dumps(task))
    for steps in [[{'operation':'tool_call','tool_name':'write','arguments':{}}],
                  ['read something'], [{'operation':'tool_call','tool_name':'read','arguments':{},'capture':{'x':'$.id'}}]]:
        with pytest.raises(ValueError,match='read-only tool calls'):
            adjudicator.replay_probe(tmp_path,{'kind':'reference_recovery','scenario_id':'reference','before_step':1,'steps':steps})
    task['business_lifecycle']={'events':[{}]}
    (tmp_path/'task.json').write_text(json.dumps(task))
    with pytest.raises(ValueError,match='static read-only'):
        adjudicator.replay_probe(tmp_path,{'kind':'reference_recovery'})
