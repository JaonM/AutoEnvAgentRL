"""Replay historical demonstrations verbatim on versioned coherent worlds, without LLM calls.

Only unchanged policy, tool observations and independently successful outcomes can
reuse the prior dialogue audit. Original evidence remains immutable; migrated
records explicitly distinguish replay from newly collected teacher trajectories.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
from pathlib import Path
from env_factory.core import spec as request_spec
import traceback

from env_factory.certify import bundle, trajectory, policy_gate
from env_factory.core import answers, reward
from env_factory.runtime import Episode, BusinessError, ToolError, run_reference
from env_factory.production import task_requirements, world
from env_factory.serving import server, agent_view
from repair_world_semantics import prepare, save
from release_diverse_dataset import validate_evidence


def read(p): return request_spec.read(p) if Path(p).name == 'request.json' else json.loads(Path(p).read_text())


def replay(loaded, old):
    request=loaded['request'];facts=copy.deepcopy(request['user_facts']);env=loaded['env']
    api=Episode(env.make_tools(),request['tables'],facts,start=request['time']['start'],
                approve=lambda tool,args:facts.get(env.approval_key(tool,args,copy.deepcopy(facts))) is True)
    calls={};seen=set();observations=[]
    for index,message in enumerate(old['messages']):
        if message.get('role')=='assistant':
            for call in message.get('tool_calls') or []:
                if call['id'] in calls:raise ValueError('duplicate tool call id')
                calls[call['id']]=call['function']
        elif message.get('role')=='tool':
            cid=message['tool_call_id']
            if cid in seen or cid not in calls:raise ValueError('unmatched tool response')
            seen.add(cid);call=calls[cid]
            if message.get('name')!=call['name']:raise ValueError('tool name mismatch')
            args=call['arguments'];args=json.loads(args) if isinstance(args,str) else args
            try:
                result=api.call(call['name'],**args)
                actual={'ok':True,'result':result,'now':api.now}
            except BusinessError as exc:actual={'ok':False,'kind':'business','error':str(exc),'now':api.now}
            except ToolError as exc:actual={'ok':False,'kind':'tool_error','error':str(exc),'now':api.now}
            expected=json.loads(message['content'])
            observations.append({'message':index,'tool':call['name'],'identical':actual==expected})
            if actual!=expected:
                return {'passed':False,'reason':'tool_observation_changed','message':index,
                        'tool':call['name'],'expected':expected,'actual':actual,'observations':observations}
    if seen!=set(calls):raise ValueError('missing tool response')
    api.submit(old['answer'])
    ref=run_reference(env,request['tables'],facts,start=request['time']['start'])
    expected=loaded['verifier'].expected(copy.deepcopy(request['tables']),copy.deepcopy(facts),request['time']['start'])
    grade=reward.grade(api.result(),expected,loaded['schema'],initial=request['tables'],reference_calls=ref['tool_calls'],
                       known=answers.known_strings(request,facts));grade.pop('state_diff',None)
    return {'passed':grade['success'],'reason':'verified_replay' if grade['success'] else 'new_verifier_rejected',
            'grade':grade,'observations':observations}


def migrate(result, worlds, output):
    group=result['domain'];assignment=result['assignment'];folder=output/group/'tasks'/f"t{assignment['index']:03d}"
    receipt=folder/'migration.json'
    if receipt.exists():return read(receipt)
    folder.mkdir(parents=True,exist_ok=True)
    row={'domain':group,'assignment':assignment,'slot':result['slot'],'passed':False,
         'original_episode':result['collection']['selected_episode']}
    old=read(row['original_episode']);candidate=read(old['source_episode'])
    oldpilot=read(Path(old['source_episode']).parent/'pilot_result.json')
    domain=old['domain'];contract=read(worlds/domain/'behavior_contract.json')
    try:
        profile=read(worlds/'summary.json').get('semantic_profile')
        newbundle=prepare(oldpilot['bundle'],folder/'semantic_task',domain,contract,sop=profile=='sop-v1')
        loaded=bundle.load_bundle(newbundle);request=loaded['request']
        original_request=read(Path(oldpilot['bundle'])/'request.json')
        if request['agent_policy']!=original_request['agent_policy']:raise ValueError('public policy changed: new audit needed')
        ref=run_reference(loaded['env'],request['tables'],request['user_facts'],start=request['time']['start'])
        quality=task_requirements.evaluate(loaded['env'],request,assignment['requirements'])
        sensitivity=world.goal_sensitivity(loaded['verifier'],request)
        statuses=[v.get('status') for v in ref['answer'].values() if isinstance(v,dict)]
        pilot={**oldpilot,'bundle':newbundle,'reference_answer':ref['answer'],'reference_trace':ref['trace'],
               'reference_statuses':statuses,'task_quality':quality,'sensitivity':sensitivity,
               'assignment_outcomes_passed':set(assignment.get('required_statuses',[]))<=set(statuses),
               'migration':{'kind':'recertified_historical_task','source_pilot':str(Path(old['source_episode']).parent/'pilot_result.json')}}
        pilot['passed']=quality['passed'] and sensitivity['passed'] and pilot['assignment_outcomes_passed']
        save(folder/'pilot_result.json',pilot);row['pilot']=str((folder/'pilot_result.json').resolve())
        if not pilot['passed']:
            row.update(reason='task_requalification_failed',recollect=False);save(receipt,row);return row
        checked=replay(loaded,old);save(folder/'replay.json',checked)
        row.update(reason=checked['reason'],recollect=not checked['passed'])
        if checked['passed']:
            # Reference answer and user-visible policy are unchanged, so inherited
            # sufficiency and quote-based audit evidence retain their meaning.
            if ref['answer']!=oldpilot['reference_answer']:
                row.update(reason='reference_answer_changed',recollect=True);save(receipt,row);return row
            service=server.Service(loaded['env'],loaded['verifier'],loaded['schema'],request,perturbed=0)
            provenance={'kind':'deterministic_replay','original_episode':row['original_episode'],
                        'original_sha256':bundle.sha256(row['original_episode']),
                        'original_demonstration':old['source_episode'],'original_demonstration_sha256':bundle.sha256(old['source_episode']),
                        'messages_unchanged':True,'tool_observations_unchanged':True,'public_policy_unchanged':True,
                        'tool_documentation_refreshed':True,'new_teacher_collection':False,
                        'audit_reuse_basis':'identical dialogue, observations, public policy and reference answer; evidence revalidated'}
            target=folder/'demonstration-replayed.json'
            selected={**old,**checked['grade'],'source_episode':str(target.resolve()),'sandbox':newbundle,
                      'platform_identity':bundle.platform_identity(),'evaluation_identity':policy_gate.evaluation_identity(),
                      'tools':agent_view.tools(service.plain_instructions(),request['category'],answer_tool=False),'migration':provenance}
            selected=trajectory.qualify(selected,trajectory.constraints(request))
            migrated={**candidate,'episode':selected,'migration':provenance}
            validate_evidence(selected,migrated,pilot,loaded)
            save(target,migrated);save(folder/'selected_episode.json',selected)
            collection={'passed':True,'selected_episode':str((folder/'selected_episode.json').resolve()),'bundle':newbundle,
                        'domain':group,'slot':assignment['slot'],'index':assignment['index'],'training_ready':False}
            save(folder/'collection.json',collection);row.update(passed=True,collection=collection)
    except Exception as exc:
        row.update(reason='migration_error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),recollect=False)
    save(receipt,row);return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--progress',type=Path,required=True);p.add_argument('--worlds',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--parallel',type=int,default=4);p.add_argument('--limit',type=int)
    args=p.parse_args();args.output=args.output.resolve();args.worlds=args.worlds.resolve()
    rows=[r for r in read(args.progress)['results'] if r['passed']]
    if args.limit:rows=rows[:args.limit]
    results=[]
    with ProcessPoolExecutor(args.parallel) as pool:
        futures=[pool.submit(migrate,r,args.worlds,args.output) for r in rows]
        for f in as_completed(futures):
            results.append(f.result());summary={'phase':'completed' if len(results)==len(rows) else 'replaying',
                'total':len(rows),'completed':len(results),'reused':sum(r['passed'] for r in results),
                'recollect':sum(r.get('recollect',False) for r in results),
                'reasons':dict(Counter(r['reason'] for r in results)),'results':results,'training_ready':False}
            save(args.output/'progress.json',summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
