#!/usr/bin/env python3
"""Paired model experiment on frozen failed sources, never fresh-generation yield.

Each author gets the same source, guide and cumulative time budget, at most two
turns, and feedback from the same compiler/reviewer. Credentials stay in the
parent evaluator; authors edit only their workspace. Existing trials are skipped
only when a terminal result exists; do not start a second controller concurrently.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'src'))
from env_factory.generation.agent_cli import command, completed_events
from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.delivery_preflight import verify_delivery
from env_factory.generation.semantic_review import review_source, resolve_privacy_review, SemanticReviewUnavailable
from env_factory.generation.artifacts import write_task_artifact
from env_factory.tasks.task import Task
from env_factory.graph.knowledge_graph import TaskType
from scripts.sandbox.generate_sandbox_scaffold import generate
from scripts.experiments.summarize_fixed_repairs import network_interrupted


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def platform_hash():
    files = [* (PROJECT/'src/env_factory').rglob('*.py'),
             * (PROJECT/'scripts/sandbox').glob('*.py'), PROJECT/'docs/code_agent_authoring.md']
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(str(p.relative_to(PROJECT)).encode()+b'\0'+p.read_bytes())
    return h.hexdigest()


def author_environment():
    return {k:v for k,v in os.environ.items()
            if 'API_KEY' not in k and not k.endswith('TOKEN') and not k.startswith('SANDBOX_')}


def run_author(model, workspace, prompt, attempt, remaining):
    started = time.monotonic()
    (attempt/'prompt.txt').write_text(prompt)
    with (attempt/'events.jsonl').open('w') as out, (attempt/'stderr.log').open('w') as err:
        p = subprocess.Popen(command('codex', model, prompt, attempt/'response.txt'),
            cwd=workspace, stdin=subprocess.PIPE, stdout=out, stderr=err,
            text=True, start_new_session=True, env=author_environment())
        try:
            p.communicate(prompt, timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            if network_interrupted(attempt/'events.jsonl'):
                raise RuntimeError('author interrupted by network outage; see events.jsonl')
            raise TimeoutError('cumulative author budget exhausted')
    if p.returncode:
        raise RuntimeError(f'author CLI exit {p.returncode}; see stderr.log')
    turns = completed_events('codex', (attempt/'events.jsonl').read_text(), attempt/'response.txt')
    if not turns: raise RuntimeError('author returned no completed turn')
    usage = {}
    for turn in turns:
        for k,v in turn.get('usage', {}).items():
            if isinstance(v,(int,float)) and not isinstance(v,bool): usage[k]=usage.get(k,0)+v
    return {'seconds':time.monotonic()-started,'usage':usage,'completed_turns':len(turns),
            'events_sha256':digest(attempt/'events.jsonl')}


def docker_check(artifacts, compiled_root, trial, tag, probe_path):
    root=compiled_root.parent/'container';root.mkdir()
    shutil.copytree(compiled_root/'data',root/'data')
    task=write_task_artifact(root,Task(artifacts['task'],artifacts['environment'],artifacts['metrics'],
        task_type=TaskType(artifacts['task_type']),task_intent=artifacts['task_intent'],
        complexity=artifacts['complexity'],artifacts=artifacts),artifacts['training_category'])
    contract=json.loads(task.read_text());contract.pop('actions',None);save(root/'BUILD_CONTRACT.json',contract)
    generate(root)
    for name in ['sandbox_runtime.py','runtime_llm.py']:
        shutil.copy2(PROJECT/'src/env_factory'/name,root/name)
    with (trial/'docker_build.log').open('w') as log:
        subprocess.run(['docker','build','-t',tag,str(root)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    env=dict(os.environ);env['SANDBOX_TRAINER_API_KEY']='paired-synthetic-trial'
    cmd=['docker','run','--rm','-i','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges:true',
         '--cpus=1','--memory=512m','--pids-limit=128','--tmpfs','/tmp:rw,nosuid,size=64m',
         '--tmpfs','/app/.runtime:rw,nosuid,uid=10001,gid=10001,size=64m']
    for name in ['SANDBOX_LLM_API_KEY','SANDBOX_LLM_BASE_URL','SANDBOX_LLM_MODEL','SANDBOX_EVALUATOR_MOCK','SANDBOX_TRAINER_API_KEY']:
        cmd+=['--env',name]
    cmd+=['--entrypoint','python3',tag,'-']
    done=subprocess.run(cmd,input=probe_path.read_text(),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=300)
    (trial/'docker_http.log').write_text(done.stdout+done.stderr)
    if done.returncode: raise RuntimeError('container HTTP replay failed; see docker_http.log')
    report=json.loads(done.stdout.strip().splitlines()[-1])
    report['image_id']=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',tag],text=True).strip()
    return report



def counterexample_check(case, source, request, root):
    """Predeclared changed-data controls; do not infer success from arbitrary errors."""
    candidate=copy.deepcopy(source)
    tables={t['table_name']:t for t in candidate['tables']}
    negative=False
    if case=='guarded_transition':
        row=next(r for r in tables['artifacts']['rows'] if r['artifact_id']=='A-103')
        previous=row['title']
        row['title']='北伐战争行军路线图复制件'
        # This is a positive control: the correct policy must report the new
        # observed title. Keeping its old answer would test a stale answer,
        # unfairly failing candidates that correctly score report accuracy.
        reference=candidate.get('reference', {})
        if isinstance(reference.get('answer'),str):
            reference['answer']=reference['answer'].replace(previous,row['title'])
        for scenario in candidate.get('scenarios',[]):
            if scenario.get('kind')=='goal_success':
                for step in scenario.get('steps',[]):
                    if step.get('operation')=='agent_response' and isinstance(step.get('content'),str):
                        step['content']=step['content'].replace(previous,row['title'])
        name='descriptive_title_rename'
    elif case=='repeated_reads_recovery':
        row=next(r for r in tables['inventory']['rows'] if r['sku']=='SD-02')
        row['available']=15
        name='initial_stock_before_absolute_event'
    elif case=='reward_goal_drift':
        event=next(e for e in candidate['business_lifecycle']['dynamics']['events']
                   if e['id']=='storm_tile_allocation')
        change=next(c for c in event['changes'] if c['table']=='materials')
        change['values']['available']=10
        name='insufficient_stock_at_actual_decision';negative=True
    else:
        return {'status':'not_applicable','scope':'no additional changed-data control; standard ablations apply'}
    root.mkdir(parents=True,exist_ok=True)
    try:
        compile_source(candidate,root=root,request=request,defer_privacy_review=True)
        return {'name':name,'passed':not negative,'observation':'original success trajectory completed'}
    except Exception as exc:
        diagnostic=root/'preflight_failure.json'
        details=json.loads(diagnostic.read_text()) if diagnostic.exists() else {}
        status=details.get('details',{}).get('actual_status')
        # A concrete client rejection at execution is evidence. Schema/compiler
        # failure or an arbitrary exception is never a successful negative.
        rejected=negative and isinstance(status,int) and 400<=status<500
        return {'name':name,'passed':rejected,'observation':'client_rejection' if rejected else 'failed_or_inconclusive',
                'actual_status':status,'error':str(exc),'diagnostic':str(diagnostic) if diagnostic.exists() else None}


def trial_run(args, case, model, frozen_hash, output):
    trial=output/case['id']/model;trial.mkdir(parents=True,exist_ok=True)
    result_path=trial/'result.json'
    if result_path.exists():return json.loads(result_path.read_text())
    workspace=trial/'authoring'
    if workspace.exists():raise RuntimeError(f'unfinished trial exists: {workspace}; inspect controller before resuming')
    workspace.mkdir()
    original=args.fixed/case['id']
    for name,expected in case['files'].items():
        if digest(original/name)!=expected:raise RuntimeError('fixed input changed: '+str(original/name))
    source=json.loads((original/'source.json').read_text())
    request=json.loads((original/'request.json').read_text());request.pop('code_agent_model',None)
    save(workspace/'source.json',source);save(workspace/'request.json',request)
    shutil.copy2(PROJECT/'docs/code_agent_authoring.md',workspace/'AUTHORING.md')
    protected={name:digest(workspace/name) for name in ['request.json','AUTHORING.md']}
    prompt=f'''Repair the supplied failed Agentic RL environment in source.json.
Keep the same public task, graph, business records, user decisions and all required
outcomes, including semantic explanations, external events and fault recovery.
Do not replace the task, remove distractors, weaken rewards or edit platform code.
Read request.json and AUTHORING.md. Prefer shared business_contract targets and
platform-owned guards, atomic updates, idempotency and query-time snapshots.
The preserved public description is authoritative when old reward/goal definitions conflict.
Failure under investigation: {case['requirement']}
Write source.json and helper files only in this directory. Never inspect other
experiment trials, saved repairs, .env, credentials, or use the network. You may
read platform implementation under {PROJECT/'src/env_factory'} and run the existing
compiler. Bare python is unavailable; use {sys.executable}.
For deterministic tasks, run complete local compilation and reference checks:
PYTHONPATH={PROJECT/'src'} {sys.executable} -m env_factory.generation.agent_authoring --source source.json --request request.json --output preview
For semantic outcomes, use --structural-preview; the parent supplies the real judge.
Do not remove semantic outcomes because the author environment has no judge key.
You have a cumulative {args.author_seconds:g}-second author budget and at most one
parent-feedback repair. Keep the original public_input and task text unchanged.
Finish by stating the actual checks you ran. The parent evaluates the source independently.
'''
    result={'case':case['id'],'model':model,'scope':'fixed_failure_repair_not_fresh_generation',
            'passed':False,'execution_passed':False,'review_passed':False,'docker_passed':False,
            'attempts':[],'author_seconds':0,'usage':{},'original_source_sha256':case['files']['source.json']}
    started=time.monotonic();feedback=''
    for index in range(1,3):
        attempt=trial/f'attempt-{index}';attempt.mkdir()
        entry={'index':index};result['attempts'].append(entry)
        before=time.monotonic()
        result.update(execution_passed=False,review_passed=False,docker_passed=False)
        try:
            remaining=args.author_seconds-result['author_seconds']
            if remaining<=0:raise TimeoutError('author budget exhausted')
            author=run_author(model,workspace,prompt+feedback,attempt,remaining)
            entry['author']=author
            for k,v in author['usage'].items():result['usage'][k]=result['usage'].get(k,0)+v
        except Exception as exc:
            entry.update(status='author_timeout' if isinstance(exc,TimeoutError) else 'infrastructure_error',error=str(exc));result['usage_incomplete']=True;break
        finally:
            result['author_seconds']+=time.monotonic()-before
        try:
            if platform_hash()!=frozen_hash:raise RuntimeError('platform changed during frozen comparison')
            if any(digest(workspace/name)!=value for name,value in protected.items()):raise ValueError('protected inputs modified')
            candidate=json.loads((workspace/'source.json').read_text())
            for field in ['task','public_input']:
                if candidate.get('description',{}).get(field)!=source.get('description',{}).get(field):
                    raise ValueError('original public task was modified: '+field)
            shutil.copy2(workspace/'source.json',attempt/'source.json')
            compiled=attempt/'compiled'
            a=compile_source(candidate,root=compiled,request=request,defer_privacy_review=True)
            entry['execution_passed']=result['execution_passed']=True
            control=counterexample_check(case['id'],candidate,request,attempt/'counterexample')
            entry['counterexample']=control
            if control.get('passed') is False:raise ValueError('changed-data counterexample failed: '+json.dumps(control,ensure_ascii=False))
            review=review_source(source=candidate,artifacts=a,request=request,root=attempt/'review',
                model=args.reviewer,timeout=args.review_seconds,goal_anchor=source.get('description'))
            entry['review']=review
            if review['status']!='pass':raise ValueError('independent review failed: '+json.dumps(review['findings'],ensure_ascii=False))
            resolve_privacy_review(a,review)
            a['generation_pipeline']['verification']['source_semantics']=review
            result['review_passed']=True
            entry['delivery']=verify_delivery(a,compiled)
            save(compiled/'compiled.json',a)
            result['docker']=docker_check(a,compiled,trial,'autoenv-paired-'+case['id'].replace('_','-')+'-'+model,args.http_probe)
            result['docker_passed']=result['docker']['passed']
            result['passed']=result['docker_passed'];entry['status']='passed';break
        except Exception as exc:
            entry.update(status='infrastructure_error' if isinstance(exc,SemanticReviewUnavailable) else 'validation_failed',error_type=type(exc).__name__,error=str(exc))
            if isinstance(exc,SemanticReviewUnavailable):break
            save(workspace/'parent_feedback.json',entry)
            feedback='\nThe parent rejected the previous source. Read parent_feedback.json, verify its evidence, and make the one allowed focused repair. Preserve the original public task. Remaining author budget: '+str(max(0,args.author_seconds-result['author_seconds']))+' seconds.\n'
        finally:
            save(trial/'progress.json',result)
    result['seconds']=time.monotonic()-started
    result['platform_unchanged']=platform_hash()==frozen_hash
    result['final_source_sha256']=digest(workspace/'source.json')
    result['training_ready']=False
    result['certification_scope']='reference_and_docker_delivery_only; live_policy_certification_pending'
    save(result_path,result)
    print(json.dumps({k:result[k] for k in ['case','model','passed','execution_passed','review_passed','docker_passed','seconds','usage']},ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--http-probe',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',help='Subset of frozen cases; original order is preserved')
    parser.add_argument('--models',nargs=2,default=['gpt-6-luna','gpt-6-astra'])
    parser.add_argument('--author-seconds',type=float,default=600)
    parser.add_argument('--reviewer',default='gpt-6-astra')
    parser.add_argument('--review-seconds',type=float,default=240)
    args=parser.parse_args();args.fixed=args.fixed.resolve();args.output=args.output.resolve();args.http_probe=args.http_probe.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    from dotenv import dotenv_values
    env=dotenv_values(PROJECT/'.env')
    os.environ.update(SANDBOX_LLM_API_KEY=env.get('SANDBOX_LLM_API_KEY') or env.get('LLM_API_KEY') or '',
        SANDBOX_LLM_BASE_URL='https://api.deepseek.com',SANDBOX_LLM_MODEL='deepseek-flash',SANDBOX_EVALUATOR_MOCK='0')
    cases=json.loads((args.fixed/'manifest.json').read_text())['cases'];frozen=platform_hash()
    if args.cases and set(args.cases)-{c['id'] for c in cases}:raise ValueError('unknown frozen case')
    plan={'models':args.models,'cases':cases,'author_seconds':args.author_seconds,
        'max_author_turns':2,'reviewer':args.reviewer,'review_seconds':args.review_seconds,
        'controller_sha256':digest(Path(__file__)),
        'platform_sha256':frozen,'guide_sha256':digest(PROJECT/'docs/code_agent_authoring.md'),
        'selected_cases':args.cases,
        'counterexamples':['descriptive_title_rename','initial_stock_before_absolute_event','insufficient_stock_at_actual_decision'],
        'scope':'paired_fixed_failure_repair_not_fresh_yield_or_training_certification'}
    previous=args.output/'plan.json'
    if previous.exists() and json.loads(previous.read_text())!=plan:raise RuntimeError('experiment settings/platform changed; use a new output root')
    save(previous,plan)
    results=[]
    for i,case in enumerate(cases):
        if args.cases and case['id'] not in args.cases:continue
        # Alternate the first model to reduce fixed ordering effects.
        for model in args.models if i%2==0 else list(reversed(args.models)):
            results.append(trial_run(args,case,model,frozen,args.output))
            save(args.output/'results.json',results)
            final=results[-1]['attempts'][-1]
            if final.get('status')=='infrastructure_error' or final.get('error_type')=='SemanticReviewUnavailable':
                print('Infrastructure unavailable; remaining trials were not started.',flush=True)
                return 2
    print('Comparison finished.',flush=True)

if __name__=='__main__':raise SystemExit(main())
