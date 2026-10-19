"""Preserve certified replay history, then sample goal-driven SOP methods for deficits."""
import argparse
from collections import Counter
import copy
from pathlib import Path
import subprocess
import sys

from prepare_scaled_experiment import read, save
from world_extensions.sop_specs import SAMPLING_PREDICATES


def assignment(base, name, method, schema):
    result=copy.deepcopy(base)
    refused=any('refused' in values for values in method.get('statuses',{}).values())
    writes=sum(bool(schema[p].get('approval')) and 'refused' not in method.get('statuses',{}).get(str(i),[])
               for i,p in enumerate(method['processes']))
    result.update(processes=method['processes'],type='sop_composition',shape='complex',sop_method=name,
        requirements={'min_goals':len(method['processes']),'min_writes':writes,'state_effect':'changed','sop_method':name},
        required_statuses=sorted({s for values in method.get('statuses',{}).values() for s in values}),
        script=['multi_goal','fallback'] if refused else ['multi_goal'],
        quota_group='refusal_partial' if refused else 'multi_write' if writes>=2 else 'single_write',
        coverage_branch='sop:'+name,coverage_predicates=copy.deepcopy(SAMPLING_PREDICATES.get(name,[])),coverage_obstacle='fallbacks' if refused else 'verified_script',
        brief=f'Use the declared SOP method {name}: '+method['intent']+
            ' Follow its shared-object constraints, actual runtime bindings and outcome branches exactly. '
            'Each write needs separate customer confirmation. The opening gives only the initial request; '
            'later needs arise according to the user script. Never place future IDs, policy answers, graph syntax '
            'or an oracle result in the user opening. Vary policy conditions, object multiplicity and customer '
            'interaction; changing names, IDs and amounts alone is not diversity. '+
            'Narrative family: '+str(base.get('scenario_family','customer follow-up'))+'.')
    return result


def adapt(plan):
    designs={g['name']:read(Path(g['bundle'])/'request.json') for g in plan['domains']}
    counts=Counter();jobs=plan['jobs']
    for job in jobs:
        if job.get('prepared_task'):continue
        group=job['domain']['name'];world=designs[group];methods=world['sop_blueprint']['methods']
        def convert(a):
            if len(a['processes'])<2:return a
            name=a.get('sop_method') or a.get('requirements',{}).get('sop_method')
            if name not in methods:
                names=list(methods)
                name=names[counts[group]%len(names)];counts[group]+=1
            return assignment(a,name,methods[name],world['goal_schema'])
        job['assignment']=convert(job['assignment'])
        job['alternatives']=[convert(a) for a in job.get('alternatives',[job['assignment']])]
    # Allocate explicit attempts for every method in each split that still needs
    # samples; this is a coverage target, never a claim of achieved acceptance.
    priority=[]
    for group,world in designs.items():
        available=[j for j in jobs if j['domain']['name']==group and not j.get('prepared_task')]
        for (name,method),job in zip(world['sop_blueprint']['methods'].items(),available):
            a=assignment(job['assignment'],name,method,world['goal_schema'])
            job.update(assignment=a,alternatives=[copy.deepcopy(a) for _ in range(3)])
            priority.append(job)
    priority_ids={id(j) for j in priority}
    plan['jobs']=priority+[j for j in jobs if id(j) not in priority_ids]
    for i,job in enumerate(plan['jobs']):
        if job.get('prepared_task'):continue
        for a in [job['assignment'],*job.get('alternatives',[])]:a.update(index=300000+i,slot=300000+i)
    plan.update(semantic_profile='sop-v1',sop_required_for_new_tasks=True,
        planned_sop_methods={g:list(w['sop_blueprint']['methods']) for g,w in designs.items()},
        method_candidates=dict(Counter(j['assignment'].get('sop_method','single') for j in plan['jobs'])),
        limits=['Only accepted samples count toward 1000 train / 200 dev; no failed task can release training.',
                'Historical dialogues are retained only with verified replay provenance.',
                'Declared methods are a finite initial catalogue, not exhaustive benchmark coverage.'])
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replay','worlds','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_semantic_experiment.py')),
        '--replay',str(args.replay),'--worlds',str(args.worlds),'--output',str(args.output)],check=True)
    path=args.output/'fill_plan/plan.json';plan=adapt(read(path));save(path,plan)
    save(args.output/'sop_plan_summary.json',{k:plan[k] for k in ('initial_counts','initial_deficits',
        'method_candidates','planned_sop_methods','semantic_profile')})


if __name__=='__main__':main()
