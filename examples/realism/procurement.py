"""Materialize a reproducible procurement/cancellation sandbox using the shared scaffold.

This is an executable runtime example, not a model-generated training-ready claim.
Run: python examples/realism/procurement.py --output output/procurement-realism
"""
import argparse
import json
from pathlib import Path
import shutil
from scripts.sandbox.generate_sandbox_scaffold import generate


def materialize(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    data=root/'data';data.mkdir(exist_ok=True)
    tables={'inventory':{'id':1,'status':'available','available':50,'reserved':0},
            'orders':{'id':1,'status':'draft'},'payments':{'id':1,'status':'unpaid'}}
    manifest={'environment_mode':'stateful','tables':[]}
    for name,row in tables.items():
        schema={'columns':[{'name':key,'type':'integer' if type(value) is int else 'text','nullable':False}
                           for key,value in row.items()],'primary_key':['id']}
        (data/f'{name}.schema.json').write_text(json.dumps(schema))
        (data/f'{name}.jsonl').write_text(json.dumps(row)+'\n')
        manifest['tables'].append({'table_name':name,'schema_file':f'{name}.schema.json','rows_file':f'{name}.jsonl'})
    tools=[];impl=[];policies={}
    for table in tables:
        name='read_'+table
        tools.append({'type':'function','function':{'name':name,'description':'Read current '+table,
            'parameters':{'type':'object','properties':{'id':{'type':'integer'}},'required':['id'],'additionalProperties':False}}})
        impl.append({'tool_name':name,'operation':'select','table':table,'filters':[{'column':'id','argument':'id','operator':'eq'}],
                     'projection':list(tables[table]),'result_field':'records'})
    operations=[('reserve','inventory','available','reserved',{'available':20,'reserved':20},'purchase'),
                ('place_order','orders','draft','placed',{},'purchase'),
                ('pay','payments','unpaid','paid',{},'purchase'),
                ('cancel_order','orders','placed','cancelled',{},'cancel'),
                ('refund','payments','paid','refunded',{},'cancel'),
                ('release','inventory','reserved','available',{'available':40,'reserved':0},'cancel')]
    for name,table,before,after,values,grant in operations:
        values={'status':after,**values}
        properties={'id':{'type':'integer'},**{key:{'type':'integer' if type(value) is int else 'string','enum':[value]} for key,value in values.items()}}
        tools.append({'type':'function','function':{'name':name,'description':f'{name}: {before} -> {after}; requires user authorization and Idempotency-Key.',
            'parameters':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}})
        impl.append({'tool_name':name,'operation':'update','table':table,'selector':{'id':'id'},
                     'changes':{key:key for key in values},'result_field':'updated_count'})
        policies[name]={'roles':['buyer'],'requires_grants':[grant],'require_idempotency':True,
            'transition':{'table':table,'selector':{'id':{'argument':'id'}},'column':'status','from':[before],'to':after}}
    stage={'id':'approval','kind':'execution_confirmation','purpose':'Confirm purchase authorization',
        'assistant_contains_all':['approve'],'reference_response':'Please approve one order of 20 units.',
        'user_reply':'I approve one order of 20 units.','retry_reply':'Ask for approval first.',
        'before_tool':'reserve','requires_tools':['read_inventory'],'grant_permissions':['purchase'],
        'branches':[{'id':'reject_split','when':[{'source':'assistant_text','path':'$','operator':'contains','expected':'split'}],
            'user_reply':'No split orders; propose one order.','outcome':'user_rejection','advance':False,
            'grant_permissions':[],'revoke_permissions':['purchase']}]}
    cancellation={'id':'cancel_request','kind':'execution_confirmation','purpose':'User cancels after payment',
        'assistant_contains_all':['payment'],'reference_response':'Payment is confirmed. Any changes?',
        'user_reply':'The event is cancelled. Cancel the order, refund payment and release the reservation.',
        'retry_reply':'Please confirm payment status first.','before_tool':'cancel_order','requires_tools':['pay'],
        'grant_permissions':['cancel'],'revoke_permissions':['purchase']}
    profile=[{'profile_id':'buyer'}]
    scripts=[{'script_id':'cancellation','initial_state':'approval','states':[],'transitions':[],
        'interaction_protocol':{'version':'1.0','stages':[stage,cancellation]}}]
    (data/'profiles.json').write_text(json.dumps(profile));(data/'scripts.json').write_text(json.dumps(scripts))
    predicates=[{'table':'orders','where':{'id':1},'values':{'status':'cancelled'},'count':1},
                {'table':'payments','where':{'id':1},'values':{'status':'refunded'},'count':1},
                {'table':'inventory','where':{'id':1},'values':{'reserved':0,'status':'available'},'count':1,
                 'value_expressions':{'available':{'at_tool':{'name':'read_inventory','expression':{
                     'lookup':{'table':'inventory','field':'available','where':{'id':1}}}}}}}]
    metric={'id':'restored','category':'outcome','type':'rule-based','scope':'terminal','weight':1,'score_range':[0,1]}
    contract={'task':'Procure 20 units, respond to the user, and leave no unwanted obligations.',
        'training_category':'multi_step_agentic','training_contract':{'category':'multi_step_agentic','sandbox_profile':'dependent_tool_chain'},
        'public_input':{'initial_user_message':'Reserve 20 units and arrange one order after my approval. Confirm payment with me.','materials':[]},
        'user_simulation_policy':{'mode':'interactive_goal'},'tools':tools,'tool_implementations':impl,'metrics':[metric],
        'metric_implementations':[{'metric_id':'restored','source':'business_state','path':'$','operator':'state_predicates',
            'expected':predicates,'score_mapping':{'pass':1,'fail':0}}],
        'task_spec':{'training_contract':{'category':'multi_step_agentic'},'requires_user_interaction':True,
            'capability_dag':{'nodes':['read_inventory','reserve','place_order','pay','cancel_order','refund','release'],
                'edges':[{'from_tool':'read_inventory','to_tool':'reserve','result_path':'$.records[0].id','argument_path':'$.id'}]},
            'goal_contract':{'row_predicates':predicates,'requires_state_change':True,
                'table_primary_keys':{name:['id'] for name in tables}}},
        'business_lifecycle':{'version':'1.0','roles':['buyer'],'tools':policies,'dynamics':{
            'tick_seconds':30,'events':[{'id':'competing_reservation','at_tick':1,'changes':[{
                'table':'inventory','selector':{'id':1},'values':{'available':40}}]}],
            'faults':[{'id':'lost_order_response','tool':'place_order','call_index':1,'phase':'after_commit','status':504},
                      {'id':'payment_unavailable','tool':'pay','call_index':1,'phase':'before','status':503}]}},
        'artifacts':{'data_manifest':{'root':'data',**manifest},'user_simulation_manifest':{
            'root':'data','profiles_file':'profiles.json','scripts_file':'scripts.json'}}}
    (root/'BUILD_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    generate(root)
    project=Path(__file__).resolve().parents[2]
    for name in ('sandbox_runtime.py','runtime_llm.py'):
        shutil.copyfile(project/'src/env_factory'/name,root/name)
    # Standard library runtime: no package installation or model credentials needed.
    (root/'Dockerfile').write_text('ARG BASE_IMAGE=python:3.12-slim\nFROM ${BASE_IMAGE}\nWORKDIR /app\nCOPY . .\nCMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080"]\n')
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/procurement-realism'))
    print(materialize(parser.parse_args().output))
