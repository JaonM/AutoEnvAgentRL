"""Preserve verified replay baselines, recollect changed observations, fill diverse deficits."""
import argparse
from collections import Counter, defaultdict, deque
import copy
from pathlib import Path
import subprocess
import sys

from env_factory.certify import bundle
from env_factory.diversity import features, Normalizer, digest, AdmissionLedger
from env_factory.diversity_context import signature
from prepare_scaled_experiment import read, save


def prepare(replay_root, preparation):
    replay=read(replay_root/'progress.json')
    if replay['phase']!='completed' or replay['completed']!=replay['total']:
        raise ValueError('historical replay is incomplete')
    source=read(preparation/'source_plan.json')
    source['scope']='Coherent worlds: verified historical replay plus newly collected original tasks; no benchmark tasks.'
    save(preparation/'source_plan.json',source)
    planpath=preparation/'fill_plan/plan.json';plan=read(planpath)
    rules=read(plan['context_rules']);ledger=AdmissionLedger([],{'train':3,'dev':1})
    baseline=[];excluded=[];recollect=[]
    groups={j['domain']['name']:j['domain'] for j in plan['jobs']}
    # Stable order, with the stricter dev cap first; completion timing cannot change selection.
    rows=sorted(replay['results'],key=lambda r:(not r['domain'].endswith('_dev'),r['domain'],r['slot']))
    for row in rows:
        if not row['passed']:
            if row.get('recollect'):
                pilot=read(row['pilot']);assignment=copy.deepcopy(row['assignment'])
                recollect.append({'domain':groups[row['domain']],'assignment':assignment,'task_attempts':1,
                    'prepared_task':{'pilot':row['pilot'],'pilot_sha256':bundle.sha256(row['pilot']),
                                     'manifest_sha256':bundle.sha256(Path(pilot['bundle'])/'manifest.json')},
                    'migration_reason':row['reason']})
            else:excluded.append({'domain':row['domain'],'slot':row['slot'],'reason':row['reason']})
            continue
        ep_path=row['collection']['selected_episode'];ep=read(ep_path);pilot=read(row['pilot'])
        request=read(Path(pilot['bundle'])/'request.json')
        feature=features(request,pilot,ep)
        key,context=signature(request,pilot,feature,Normalizer(request),rules[ep['domain']],digest)
        owner=(row['domain'],row['slot']);rejection=ledger.reserve(owner,ep['split'],key)
        if rejection:
            excluded.append({'domain':row['domain'],'slot':row['slot'],'reason':rejection});continue
        ledger.finish(owner,accepted=True)
        baseline.append({**feature,'id':f"{row['domain']}:{row['slot']}",'group':row['domain'],'domain':ep['domain'],
            'split':ep['split'],'slot':row['slot'],'index':row['assignment']['index'],'bundle':pilot['bundle'],
            'episode':ep_path,'task_type':row['assignment']['type'],'contextual_task_key':key,'policy_context':context,
            'goals':[dict(process=g['process'],params=g['params'],observed=pilot['reference_answer'][f'goal_{i}'])
                     for i,g in enumerate(request['user_facts']['goals'],1)],'provenance':'deterministic_replay'})
    counts=Counter(r['group'] for r in baseline)
    audit=preparation/'replay_baseline';save(audit/'records.json',baseline);save(audit/'context_rules.json',rules)
    selection=preparation/'replay_selection'
    save(selection/'selection_manifest.json',{'kept':[{'id':r['id']} for r in baseline],'excluded':excluded})
    save(selection/'plan.json',{'jobs':[]})
    # Recompute quota deficits from admitted replays; do not restart a zero-baseline distribution.
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_diverse_fill.py')),
        '--audit',str(audit),'--selection',str(selection/'selection_manifest.json'),
        '--supplement',str(preparation/'empty_supplement'),'--source-plan',str(preparation/'source_plan.json'),
        '--output',str(preparation/'deficit_plan')],check=True)
    def matches(goal,predicates):
        for predicate in predicates:
            value=goal
            for part in predicate['path'].split('.'):
                value=value.get(part) if isinstance(value,dict) else None
            if value!=predicate['equals']:return False
        return True
    missing=defaultdict(deque)
    for job in read(preparation/'branch_plan/plan.json')['jobs']:
        a=job['assignment'];obstacle=a['coverage_obstacle']
        covered=any(r['group']==job['domain']['name'] and any(matches(g,a['coverage_predicates']) for g in r['goals'])
                    and (obstacle=='verified_script' or obstacle in r['observed_obstacles']) for r in baseline)
        if not covered:missing[job['domain']['name']].append(a)
    plan=read(preparation/'deficit_plan/plan.json');seen=Counter();branch_slots=0
    for i,job in enumerate(plan['jobs']):
        group=job['domain']['name'];seen[group]+=1
        if missing[group] and seen[group]%2:
            a=copy.deepcopy(missing[group].popleft());job['assignment']=a
            job['alternatives']=[a,copy.deepcopy(a),job['alternatives'][2]];branch_slots+=1
        for a in [job['assignment'],*job.get('alternatives',[])]:a.update(index=100000+i,slot=100000+i)
    plan.update(jobs=recollect+plan['jobs'],source_audit=str(audit.resolve()),context_rules=str((audit/'context_rules.json').resolve()),
        initial_counts=dict(counts),initial_deficits={g:n-counts[g] for g,n in plan['fill_targets'].items()},
        historical_demonstrations_reused=len(baseline),recollection_jobs=len(recollect),planned_branch_slots=branch_slots,
        fresh_world_data=False,semantic_profile='coherent-v1',historical_replay=str(replay_root.resolve()))
    save(planpath,plan)
    summary={'baseline':len(baseline),'counts':dict(counts),'recollection_jobs':len(recollect),'excluded':excluded,
             'fresh_candidate_budget':len(plan['jobs'])-len(recollect),'planned_branch_slots':branch_slots,'training_ready':False}
    save(preparation/'semantic_preparation.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--worlds',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_scaled_experiment.py')),
        '--worlds',str(args.worlds),'--output',str(args.output)],check=True)
    import json
    print(json.dumps(prepare(args.replay,args.output),ensure_ascii=False))


if __name__=='__main__':main()
