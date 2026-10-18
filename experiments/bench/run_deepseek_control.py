"""Evaluate the configured DeepSeek teacher on the frozen 164-task quick-eval population.

One trial only. Credentials remain in process environment, not CLI arguments or run metadata.
Run with .bench/tau2/bin/python; --report-only never calls a provider.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from env_factory.core.llm import load_env_file
from behavior_funnel import is_write

DOMAINS = {'airline': 50, 'retail': 114}
SIMS = ROOT/'.bench/tau2-repo/data/simulations/v3bench'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def baseline(domain):
    data=json.loads((SIMS/f'qA3m-{domain}/results.json').read_text())
    ids={s['task_id'] for s in data['simulations']}
    tasks=[t for t in data['tasks'] if t['id'] in ids]
    assert len(tasks)==DOMAINS[domain]
    return tasks

def with_reasoning(messages, convert):
    """Preserve API-returned reasoning for DeepSeek's required tool-call continuation."""
    result=convert(messages)
    assert len(result)==len(messages)
    for original, wire in zip(messages,result):
        if wire.get('role')!='assistant':
            continue
        raw=getattr(original,'raw_data',None) or {}
        choices=raw.get('choices') or []
        response_message=choices[0].get('message',{}) if choices else {}
        reasoning=response_message.get('reasoning_content')
        if reasoning is not None:
            wire['reasoning_content']=reasoning
    return result

def worker(args):
    load_env_file(ROOT/'.env')
    os.environ['OPENAI_API_KEY']=os.environ['LLM_API_KEY']
    os.environ['TAU2_DATA_DIR']=str(ROOT/'.bench/tau2-repo/data')
    import tau2.utils.llm_utils as llm
    import tau2.evaluator.evaluator_nl_assertions as nl
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain, get_tasks
    model='openai/'+os.environ['LLM_MODEL']
    shared={'api_base':os.environ['LLM_BASE_URL'],'temperature':0.0}
    # Same simulator and NL judge settings as run_tau2.sh/tau2_run.py; key is inherited.
    nl.DEFAULT_LLM_NL_ASSERTIONS=model
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS=dict(shared)
    original_convert=llm.to_litellm_messages
    llm.to_litellm_messages=lambda messages:with_reasoning(messages,original_convert)
    original_completion=llm.completion
    lock=threading.Lock()
    def recorded_completion(*a,**kw):
        response=original_completion(*a,**kw)
        message=response.choices[0].message
        meta={'at':now(),'model':response.model,'finish_reason':response.choices[0].finish_reason,
              'has_reasoning':bool(getattr(message,'reasoning_content',None)),
              'has_tools':bool(kw.get('tools')),
              'usage':{k:getattr(response.usage,k,None) for k in ('prompt_tokens','completion_tokens','total_tokens')}}
        with lock:
            with (args.out/f'{args.domain}-provider.jsonl').open('a') as f:f.write(json.dumps(meta)+'\n')
        return response
    llm.completion=recorded_completion
    tasks=baseline(args.domain)
    loaded=get_tasks(args.domain,task_ids=[t['id'] for t in tasks])
    assert {t.id:t.model_dump(mode='json') for t in loaded}=={t['id']:t for t in tasks},'task definitions changed'
    config=TextRunConfig(domain=args.domain,task_ids=[t['id'] for t in tasks],num_trials=1,
        agent='llm_agent',llm_agent=model,
        llm_args_agent={**shared,'reasoning_effort':'high','extra_body':{'thinking':{'type':'enabled'}}},
        user='user_simulator',llm_user=model,llm_args_user=dict(shared),
        max_steps=100,seed=300,max_concurrency=14,auto_resume=True,
        save_to=f'v3bench/{args.arm}-{args.domain}')
    result=run_domain(config)
    (args.out/f'{args.domain}.done').write_text(json.dumps({'finished_at':now(),'simulations':len(result.simulations)}))

def report(args):
    rows=[];protocol={};sources={};missing=[]
    for domain,n in DOMAINS.items():
        path=SIMS/f'{args.arm}-{domain}/results.json'
        if not path.exists():missing.append(domain);continue
        data=json.loads(path.read_text());tasks={t['id']:t for t in baseline(domain)}
        assert {t['id']:t for t in data['tasks']}==tasks,'task definitions differ'
        sims=data['simulations'];assert len({s['task_id'] for s in sims})==len(sims),'duplicate task'
        info=data['info'];protocol[domain]={'agent':info['agent_info']['llm'],
            'user':info['user_info']['llm'],'seed':info['seed'],'max_steps':info['max_steps'],
            'trials':info['num_trials'],'temperature_requested':info['agent_info']['llm_args'].get('temperature')}
        assert protocol[domain]['trials']==1 and protocol[domain]['seed']==300 and protocol[domain]['max_steps']==100
        if len(sims)!=n:missing.append(f'{domain}: {len(sims)}/{n}')
        for s in sims:
            infrastructure=s.get('termination_reason')=='infrastructure_error' or s.get('reward_info') is None
            rows.append({'domain':domain,'task_id':s['task_id'],'write':any(is_write(a['name']) for a in (tasks[s['task_id']].get('evaluation_criteria') or {}).get('actions') or []),
                         'infrastructure':infrastructure,'success':not infrastructure and (s['reward_info']['reward'] or 0)>=1-1e-9,
                         'termination':s.get('termination_reason')})
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    def score(chosen):
        valid=[r for r in chosen if not r['infrastructure']];success=sum(r['success'] for r in valid)
        return {'tasks':len(chosen),'graded':len(valid),'infrastructure_errors':len(chosen)-len(valid),'successes':success,
                'success_rate':success/len(valid) if valid else None}
    groups={'overall':score(rows),**{d:score([r for r in rows if r['domain']==d]) for d in DOMAINS},
            'write':score([r for r in rows if r['write']]),'nonwrite':score([r for r in rows if not r['write']])}
    observed=collections.Counter()
    for p in args.out.glob('*-provider.jsonl'):
        for line in p.read_text().splitlines():observed[json.loads(line)['model']]+=1
    result={'generated_at':now(),'complete':not missing and len(rows)==164 and not any(r['infrastructure'] for r in rows),
            'missing':missing,'scores':groups,'protocol':protocol,'observed_response_models':dict(observed),
            'source_sha256':sources,'tasks':rows,
            'limitations':['One trial per task; no pass@8.','DeepSeek agent uses thinking/high; temperature 0 is requested but ignored in thinking mode.',
                          'deepseek-flash is a mutable API alias; returned model labels may not identify a pinned snapshot.',
                          'User simulator and judge use the existing API defaults; historical baselines may precede a provider update.']}
    (args.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['DeepSeek single-trial 164-task control',f"Complete: {result['complete']}",'group | passed / graded | infrastructure']
    for k,v in groups.items():lines.append(f"{k} | {v['successes']}/{v['graded']} | {v['infrastructure_errors']}")
    (args.out/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'output/DeepSeek_tau2_164')
    p.add_argument('--arm',default='deepseek-teacher-20261017')
    p.add_argument('--domain',choices=DOMAINS)
    p.add_argument('--report-only',action='store_true');args=p.parse_args();args.out=args.out.resolve()
    os.chdir(ROOT)
    if args.domain:return worker(args)
    if args.report_only:return report(args)
    args.out.mkdir(parents=True,exist_ok=False)
    load_env_file(ROOT/'.env')
    from env_factory.production.offpeak import wait
    wait(lambda msg:print(msg,flush=True))
    manifest={'started_at':now(),'arm':args.arm,'agent_model':os.environ['LLM_MODEL'],'trials':1,
              'thinking':'enabled','reasoning_effort':'high','temperature_requested':0,'seed':300,'max_steps':100,
              'benchmark_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'.bench/tau2-repo',text=True).strip(),
              'tasks':{d:[t['id'] for t in baseline(d)] for d in DOMAINS},
              'task_sha256':{d:digest(baseline(d)) for d in DOMAINS},
              'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    jobs=[]
    for d in DOMAINS:
        log=(args.out/f'{d}.log').open('w')
        proc=subprocess.Popen([sys.executable,__file__,'--domain',d,'--arm',args.arm,'--out',str(args.out)],stdout=log,stderr=subprocess.STDOUT)
        jobs.append((d,proc,log))
    codes={}
    for d,proc,log in jobs:codes[d]=proc.wait();log.close()
    result=report(args)
    (args.out/'status.json').write_text(json.dumps({'finished_at':now(),'exit_codes':codes,'complete':result['complete']},indent=2))
    if any(codes.values()) or not result['complete']:raise SystemExit(1)

if __name__=='__main__':main()
