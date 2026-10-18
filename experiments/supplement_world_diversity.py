"""Bounded targeted generation against a frozen platform; never releases training.

Branch and diversity checks supplement, never replace, the original world,
reference, policy, spoken-consent and dialogue quality admission gates.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter, deque
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import traceback


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--dataset-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int);args=p.parse_args()
    args.plan=args.plan.resolve();args.frozen=args.frozen.resolve();args.output=args.output.resolve()
    own=Path(__file__).resolve();utility=own.parents[1]/'src/env_factory/diversity.py'
    spec=importlib.util.spec_from_file_location('supplement_diversity',utility);div=importlib.util.module_from_spec(spec);spec.loader.exec_module(div)
    context_utility=utility.with_name('diversity_context.py')
    spec=importlib.util.spec_from_file_location('supplement_context',context_utility);ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    # Pin all generation/scoring dependencies to the preserved original source.
    sys.path[:0]=[str(args.frozen/'src'),str(args.frozen/'experiments')]
    from env_factory.authoring.codex import CodexRunner, InfrastructureError
    from env_factory.core.llm import load_env_file
    from env_factory.certify import bundle
    from env_factory.production import data_scope, dialogue_audit
    from sample_world_tasks import draft_batch, certify_task
    from generate_world_dataset import collect
    load_env_file()
    plan=read(args.plan);jobs=plan['jobs'][:args.limit] if args.limit else plan['jobs']
    audit=Path(plan['source_audit']);baseline=read(audit/'records.json')
    context_rules=read(plan['context_rules'])
    for r in baseline:r['task_key']=r['contextual_task_key']
    original=read(args.dataset_identity)
    baseline_paths={str(Path(r['episode']).resolve()):sha(r['episode']) for r in baseline}
    frozen={**original['code'],**baseline_paths,str(own):sha(own),str(utility):sha(utility),str(context_utility):sha(context_utility),
            plan['context_rules']:sha(plan['context_rules']),str(args.plan):sha(args.plan),
            str(audit/'records.json'):sha(audit/'records.json')}
    def unchanged():
        if any(sha(p)!=v for p,v in frozen.items()):raise RuntimeError('Frozen generation/audit inputs changed')
        if bundle.platform_identity()!=original['platform']:raise RuntimeError('Platform identity changed')
    unchanged()
    identity={'code':frozen,'model':CodexRunner().model,'plan':str(args.plan),'jobs':len(jobs),
              'platform':original['platform'],'training_ready':False}
    if (args.output/'identity.json').exists() and read(args.output/'identity.json')!=identity:
        raise ValueError('Supplement inputs changed; use a new output')
    save(args.output/'identity.json',identity)
    for source in (own,utility,context_utility):
        target=args.output/'source_snapshot'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    runner=CodexRunner();results=[]
    ledger=div.AdmissionLedger(baseline,{s:plan['admission']['normalized_structure_cap_'+s] for s in ('train','dev')})
    counts=Counter(r['group'] for r in baseline)
    targets=plan.get('fill_targets',{})
    def full(group):return group in targets and counts[group]>=targets[group]
    def all_full():return bool(targets) and all(full(g) for g in targets)
    def state(phase='running'):
        save(args.output/'progress.json',{'phase':phase,'target':len(jobs),'completed':len(results),
            'accepted':sum(r['passed'] for r in results),'results':results,'training_ready':False,
            'dataset_counts':dict(counts),'dataset_targets':targets,
            'remaining':{g:max(0,n-counts[g]) for g,n in targets.items()}})
    def matches(record,predicate):
        value=record
        for part in predicate['path'].split('.'):
            if not isinstance(value,dict) or part not in value:return False
            value=value[part]
        return value==predicate['equals']
    def author(job,attempt):
        unchanged();domain=job['domain']
        variant=(job.get('alternatives') or [job['assignment']])[attempt%len(job.get('alternatives') or [job['assignment']])]
        a={**variant,'slot':job['assignment']['slot'],'index':job['assignment']['index']+attempt*10000}
        previous=args.output/'attempt_results'/f"{domain['name']}-{a['slot']}-{attempt-1}.json"
        if attempt and previous.exists():
            failure=read(previous)
            feedback={k:failure[k] for k in ('stage','errors','rejection_reason') if k in failure}
            feedback['task_problems']=(failure.get('pilot') or {}).get('problems',[])
            a['brief']+='\nPrevious attempt did not qualify. Address the following without changing policy or checkers: '+json.dumps(feedback)
        folder=draft_batch(domain,[a],args.output,runner)
        row=certify_task(domain,a,folder,args.output,runner)
        if not row['passed']:return {'passed':False,'stage':'original_task_gate','pilot':row,'assignment':a}
        loaded=bundle.load_bundle(row['bundle']);request=loaded['request'];limits=domain['data_scope']
        errors=data_scope.check_references(request['tables'],limits['references'])
        for table,field in limits['ownership']['tables'].items():
            if any(r[field] not in limits['owners'] for r in request['tables'][table]) or {r['id'] for r in request['tables'][table]}!=set(limits['private_row_ids'][table]):
                errors.append('private partition changed: '+table)
        targets=[{'process':g['process'],'params':g['params'],'observed':row['reference_answer'][f'goal_{i}']}
                 for i,g in enumerate(request['user_facts']['goals'],1)]
        hit=any(all(matches(t,p) for p in a.get('coverage_predicates',[])) for t in targets)
        preconditions=div.interaction_preconditions(request,a['coverage_obstacle'])
        if preconditions:return {'passed':False,'stage':'interaction_precondition','errors':preconditions,'assignment':a}
        feature=div.features(request,row,{'messages':[]})
        feature['task_key'],feature['policy_context']=ctx.signature(request,row,feature,div.Normalizer(request),context_rules[domain['business_domain']],div.digest)
        if errors or not hit:return {'passed':False,'stage':'partition_or_target_branch','errors':errors,'target_matched':hit,'assignment':a}
        return {'passed':True,'stage':'reference_target','request':request,'pilot':row,'assignment':a,'feature':feature}
    def dialogue(job,authored):
        a=authored['assignment'];domain=job['domain'];unchanged()
        result=collect(domain,a,authored['pilot'],args.output,plan['teacher_attempts'])
        if not result['passed']:return {'passed':False,'stage':'teacher_admission','collection':result,'assignment':a}
        episode=read(result['selected_episode']);candidate=read(episode['source_episode']);obstacle=a['coverage_obstacle']
        witnesses=div.observed_obstacles(authored['request'],episode)
        audit=candidate.get('audit',{})
        for ev in audit.get('raw',{}).get('script_events',[]) if audit.get('passed') else []:
            quotes=ev.get('evidence') or []
            if ev.get('observed') and quotes and all(dialogue_audit.evidence_ok(q,episode['messages']) for q in quotes):witnesses[ev['element']]=quotes
        # No label-only coverage, even if the database goal succeeded.
        if obstacle!='verified_script' and obstacle not in witnesses:return {'passed':False,'stage':'target_interaction_not_witnessed','assignment':a,'collection':result}
        feature=div.features(authored['request'],authored['pilot'],episode)
        feature['task_key'],feature['policy_context']=ctx.signature(authored['request'],authored['pilot'],feature,
            div.Normalizer(authored['request']),context_rules[domain['business_domain']],div.digest)
        return {'passed':True,'stage':'admitted','assignment':a,'collection':result,'task_key':feature['task_key'],
                'observed_obstacles':witnesses,'tool_path':feature['tool_path']}
    queue=deque((job,0) for job in jobs);pending={}
    def reject_or_retry(job,attempt,result):
        entry={'branch':result.get('assignment',job['assignment'])['coverage_branch'],'domain':job['domain']['name'],
               'slot':job['assignment']['slot'],'attempt':attempt,**result}
        save(args.output/'attempt_results'/f"{job['domain']['name']}-{job['assignment']['slot']}-{attempt}.json",entry)
        if not result['passed'] and result['stage']!='quota_filled' and attempt+1<plan['task_attempts']:
            queue.appendleft((job,attempt+1))
        else:
            results.append({k:v for k,v in entry.items() if k not in ('request','pilot','feature')})
            state()
    def pump(authors):
        while queue and not all_full() and sum(v[0]=='author' for v in pending.values())<plan['author_parallel'] \
                and len(pending)<plan['author_parallel']+plan['dialogue_parallel']:
            job,attempt=queue.popleft()
            if full(job['domain']['name']):continue
            pending[authors.submit(author,job,attempt)]=('author',job,attempt)
    def export_candidates():
        selected=[{'episode':r['episode'],'split':r['split'],'group':r['group'],'key':r['task_key']} for r in baseline]
        selected += [{'episode':r['collection']['selected_episode'],'split':r['domain'].rsplit('_',1)[1],
                      'group':r['domain'],'key':r['task_key']} for r in results if r['passed']]
        episodes=[read(r['episode']) for r in selected]
        ids=[e['id'] for e in episodes]
        if len(ids)!=len(set(ids)):raise ValueError('duplicate task IDs in candidate export')
        for split in ('train','dev'):
            with (args.output/(split+'.jsonl')).open('w') as f:
                for e in episodes:
                    if e['split']==split:f.write(json.dumps(e,ensure_ascii=False)+'\n')
        save(args.output/'candidate_manifest.json',{'records':selected,'counts':dict(counts),'targets':targets,
            'data_target_met':all_full(),'training_ready':False,'source_hashes':{s+'.jsonl':sha(args.output/(s+'.jsonl')) for s in ('train','dev')},
            'next_gate':'Joint coverage and quality review before training release; original training queue remains paused.'})
    state()
    try:
        with ThreadPoolExecutor(plan['author_parallel']) as authors,ThreadPoolExecutor(plan['dialogue_parallel']) as dialogues:
            pump(authors)
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    kind,job,attempt=pending.pop(future);value=future.result();split=job['domain']['split']
                    owner=(job['domain']['name'],job['assignment']['slot'],attempt)
                    if kind=='author' and full(job['domain']['name']):value={'passed':False,'stage':'quota_filled'}
                    if kind=='author' and value['passed']:
                        key=value['feature']['task_key'];rejected=ledger.reserve(owner,split,key)
                        if rejected:
                            value={'passed':False,'stage':'diversity_rejected','task_key':key,'rejection_reason':rejected}
                        else:
                            pending[dialogues.submit(dialogue,job,value)]=('dialogue',job,attempt)
                            pump(authors)
                            continue
                    if kind=='dialogue':
                        if full(job['domain']['name']):value={**value,'passed':False,'stage':'quota_filled'}
                        ledger.finish(owner,accepted=value['passed'])
                        if value['passed']:counts[job['domain']['name']]+=1
                    reject_or_retry(job,attempt,value)
                    pump(authors)
        export_candidates()
        state('completed' if not targets or all_full() else 'budget_exhausted')
    except Exception as exc:
        save(args.output/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'training_ready':False})
        state('failed');raise


if __name__=='__main__':main()
