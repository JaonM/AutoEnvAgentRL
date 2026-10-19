"""Bounded, independently audited natural dialogues for newly covered flows.

Acceptance fixtures only. These are never silently added to a frozen SFT release.
"""
import argparse
import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from env_factory.core import spec as request_spec

from env_factory.certify import bundle, policy_gate, sufficiency, trajectory
from env_factory.core.llm import ChatClient, load_env_file
from env_factory.production import delivery_teacher, dialogue_audit
from env_factory.production.world import task_schema, task_category
from env_factory.runtime import is_write, tool_specs
from env_factory.serving import server
from recheck_world_tasks import save


def customer_request(case):
    p=case['params']; process=case['process']
    if process=='record_query':
        return f"What is the first passenger's first name on reservation {p['record_id']}?"
    if process=='handover':
        return {'unsupported_service':'I need help arranging a hotel room for this trip. Could you transfer me to someone who can help?',
                'change_passenger_count':'Please add another passenger to my existing reservation. If that is not allowed, just explain the restriction; do not transfer me.',
                'add_insurance_after_booking':'I already booked my ticket without insurance. Please add insurance to that existing booking. If that is not allowed, just explain the restriction; do not transfer me.'}[p['request_kind']]
    s=p.get('selection',p)
    text=(f"I need an economy flight from {s['origin']} to {s['destination']} on {s['dates'][0]} for one passenger. "
          f"The ticket budget is ${s['max_fare']} excluding bags and insurance. ")
    text+=('Only nonstop initially; if none fit my budget, allow one connection. ' if s.get('fallback') else
           'Up to one connection is fine. ')
    text+='Choose the cheapest qualifying itinerary. If tied, prefer fewer stops, then earlier departure, then the lowest flight number. I can travel any time that day, arriving before midnight. '
    if process=='constrained_booking':
        if p.get('return_selection'):
            back=p['return_selection']
            text+=(f"Also choose the cheapest nonstop economy return from {back['origin']} to {back['destination']} "
                   f"on {back['dates'][0]}, with a separate ${back['max_fare']} return ticket budget. "
                   'Book both directions together as a round trip. ')
        person=p['passengers'][0]
        text+=(f"Please book it for {person['first_name']} {person['last_name']}, born {person['dob']}, "
               f"with no checked bags and no insurance. Use my saved payment method {p['payment_id']}. "
               'Please report the booking reference and actual charge.')
    else:text+='Please tell me the chosen flights, travel date and total ticket fare. Do not book yet.'
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upgrade',type=Path,required=True)
    parser.add_argument('--parallel',type=int,default=2)
    parser.add_argument('--attempts',type=int,default=2)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ids',nargs='+')
    args=parser.parse_args();load_env_file()
    source=args.upgrade/'airline/bundle';base=bundle.load_bundle(source)
    names=['query_passenger_name','search_cheapest_connection','search_fallback','search_then_book',
           'search_then_roundtrip_book','handover_unsupported_service','handover_change_passenger_count','handover_add_insurance_after_booking']
    names=args.ids or names
    destination=args.output or args.upgrade/'natural_dialogues'
    identity={'bundle_files':base['manifest']['files'],'platform':bundle.platform_identity(),
              'runner_sha256':bundle.sha256(__file__),'evaluation_identity':policy_gate.evaluation_identity(),
              'cases':{n:bundle.sha256(args.upgrade/'airline/probes'/f'{n}.json') for n in names},
              'attempts':args.attempts,'models':{role:{'model':c.model,'endpoint':c.base_url}
                  for role in ('LLM','USER_SIM','JUDGE') for c in [ChatClient(role)]}}
    stamp=destination/'identity.json'
    if stamp.exists() and json.loads(stamp.read_text())!=identity:raise ValueError('inputs changed; use a new output')
    save(stamp,identity)

    def one(name):
        folder=destination/name;selected=folder/'selection.json'
        if selected.exists():return json.loads(selected.read_text())
        case=json.loads((args.upgrade/'airline/probes'/f'{name}.json').read_text())
        loaded=copy.copy(base);request=copy.deepcopy(base['request']);facts=copy.deepcopy(case['facts'])
        uid=case['tables']['users'][0]['user_id'];facts['user_id']=uid
        speech=customer_request(case)
        if name=='handover_unsupported_service':
            facts['goals'][0]['params']['summary']='Customer needs assistance arranging a hotel room for their trip.'
        request.update(id='coverage-'+name,tables=case['tables'],user_facts=facts,
             user_scene={'situation':'I need help with my travel plans.','goal':speech,
                         'known':{'user_id':uid},'wants_to_know':['What is the result of my request?']},
             user_request=(f"Hi, my user ID is {uid}. "+speech))
        request.pop('user_script',None)
        request['category']=task_category(source,request)
        loaded.update(request=request,schema=task_schema(request['goal_schema'],facts['goals']))
        stage=folder/'source';stage.mkdir(parents=True,exist_ok=True)
        request_spec.write(request, stage / 'request.json');save(stage/'answer_schema.json',loaded['schema'])
        for filename in ('env.py','verifier.py'):shutil.copy(source/filename,stage/filename)
        quality=bundle.qualify_loaded(loaded);save(folder/'qualification.json',quality)
        if not quality['passed']:
            result={'id':name,'passed':False,'stage':'qualification'};save(selected,result);return result
        target=folder/'bundle'
        if not target.exists():bundle.export(stage,stage/'request.json',quality,target,status='qualified_acceptance_probe')
        loaded=bundle.load_bundle(target)
        sufficient=sufficiency.check(loaded,ChatClient('JUDGE'),k=2);save(folder/'sufficiency.json',sufficient)
        if not sufficient['passed']:
            result={'id':name,'passed':False,'stage':'sufficiency'};save(selected,result);return result
        functions=loaded['env'].make_tools();write_names={n for n,fn in functions.items() if is_write(fn)}
        rules=trajectory.constraints(request)
        service=bundle.service(loaded,user_mode='natural',user_client=ChatClient('USER_SIM'),perturbed=0)
        httpd=server.start(service);attempts=[]
        try:
            for index in range(args.attempts):
                path=folder/f'attempt-{index:02d}.json'
                if path.exists():row=json.loads(path.read_text())
                else:
                    teacher=delivery_teacher.DeliveryTeacher(ChatClient('LLM'))
                    episode=trajectory.qualify(policy_gate.play(f'http://127.0.0.1:{httpd.server_address[1]}',teacher,
                        'original',request['category'],max_turns=30,answer_tool=False,
                        trajectory_constraints=rules,consent='plain'),rules)
                    actual=set()
                    for msg in episode.get('messages',[]):
                        if msg.get('role')=='tool':
                            try:
                                if json.loads(msg['content']).get('ok') is True:actual.add(msg['name'])
                            except (ValueError,AttributeError):pass
                    audited=dialogue_audit.audit(ChatClient('JUDGE'),request,episode,tool_specs(functions,consent=False),write_names)
                    required=set(case['required_tools'])<=actual
                    row=dict(episode=episode,audit=audited,required_tools_passed=required,
                        passed=bool(episode['policy_aware_success'] and audited['passed'] and required),generation=teacher.report())
                    save(path,row)
                attempts.append({'index':index,'passed':row['passed'],
                     'business_success':row['episode'].get('success'),'audit_errors':row['audit'].get('errors')})
                if row['passed']:break
        finally:httpd.shutdown();httpd.server_close()
        result=dict(id=name,passed=any(r['passed'] for r in attempts),attempts=attempts,training_ready=False)
        save(selected,result);return result

    rows=[]
    with ThreadPoolExecutor(args.parallel) as pool:
        futures={pool.submit(one,name):name for name in names}
        for future in as_completed(futures):
            try:rows.append(future.result())
            except Exception as exc:rows.append({'id':futures[future],'passed':False,'error':f'{type(exc).__name__}: {exc}'})
            summary={'completed':len(rows),'total':len(names),'passed':sum(r['passed'] for r in rows),
                     'results':rows,'training_ready':False}
            save(destination/'summary.json',summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
