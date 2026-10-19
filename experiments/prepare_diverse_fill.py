"""Prepare the admitted diverse baseline and a bounded, deficit-aware candidate bank."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import itertools
import json
from pathlib import Path
import random

from env_factory.diversity import features, Normalizer, digest, AdmissionLedger
from env_factory.diversity_context import signature


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--supplement',type=Path,required=True);p.add_argument('--source-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261019)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rng=random.Random(args.seed);source=read(args.source_plan);groups={g['name']:g for g in source['domains']}
    old={r['id']:r for r in read(args.audit/'records.json')};selection=read(args.selection)
    baseline=[old[r['id']] for r in selection['kept']]
    supplemented=read(args.supplement/'progress.json')
    assert supplemented['phase']=='completed'
    rules=read(args.audit/'context_rules.json')
    for result in supplemented['results']:
        if not result['passed']:continue
        path=result['collection']['selected_episode'];ep=read(path);candidate=read(ep['source_episode'])
        assert candidate['accepted'] and candidate['audit']['passed'] and ep['policy_aware_success']
        pilot=read(Path(ep['source_episode']).parent/'pilot_result.json');request=read(Path(pilot['bundle'])/'request.json')
        f=features(request,pilot,ep);key,context=signature(request,pilot,f,Normalizer(request),rules[ep['domain']],digest)
        assert key==result['task_key']
        baseline.append(dict(id=f"{result['domain']}:{result['slot']}",group=result['domain'],domain=ep['domain'],
            split=ep['split'],slot=result['slot'],index=result['assignment']['index'],bundle=pilot['bundle'],episode=path,
            task_type='coverage_gap',contextual_task_key=key,policy_context=context,
            goals=[dict(process=g['process'],params=g['params'],observed=pilot['reference_answer'][f'goal_{i}'])
                   for i,g in enumerate(request['user_facts']['goals'],1)],**f))
    assert len(baseline)==len(selection['kept'])+supplemented['accepted']
    AdmissionLedger(baseline,{'train':3,'dev':1})
    targets={g['name']:500 if g['split']=='train' else 100 for g in groups.values()}
    counts=Counter(r['group'] for r in baseline);deficits={g:targets[g]-counts[g] for g in targets}
    save(args.output/'baseline/records.json',baseline);save(args.output/'baseline/context_rules.json',rules)
    jobs_by_group={g:[] for g in groups};category_counts=Counter()
    for r in baseline:
        ep=read(r['episode']);pilot=read(Path(ep['source_episode']).parent/'pilot_result.json')
        category=pilot['assignment'].get('quota_group','refusal_partial')
        if category=='branch_supplement':category='refusal_partial'
        category_counts[r['group'],category]+=1
    prior_plan=read(args.selection.parent/'plan.json')
    unfilled=[]
    accepted_branches={(r['domain'].split('_')[0],r['assignment']['coverage_branch']) for r in supplemented['results'] if r['passed']}
    for r in supplemented['results']:
        if r['passed'] or (r['domain'].split('_')[0],r['branch']) in accepted_branches:continue
        original=next(j for j in prior_plan['jobs'] if j['domain']['name']==r['domain'] and j['assignment']['slot']==r['slot'])
        unfilled.append(original)
    job_index=30000
    for group,g in groups.items():
        world=read(Path(g['bundle'])/'request.json');schema=world['goal_schema']
        sop_methods=(world.get('sop_blueprint') or {}).get('methods', {})
        templates={};families=sorted({a['scenario_family'] for a in g['assignments']})
        for a in g['assignments']:
            key=(a['type'],tuple(a['processes']),tuple(a['script']))
            templates.setdefault(key,a)
        by_category={q:[a for a in templates.values() if a['quota_group']==q] for q in source['quota_percent']}
        writers=[n for n,v in schema.items() if v.get('approval')]
        # Only field-free, abstract patterns from this split reach authors.
        existing=Counter(tuple(x['process'] for x in r['goals']) for r in baseline if r['group']==group)
        common=[{'processes':list(k),'count':v} for k,v in existing.most_common(6)]
        def variant(base,index,serial):
            a=copy.deepcopy(base);a.update(index=index,slot=index)
            family=families[serial%len(families)]
            a['scenario_family']=family;a['template_id']=f'diverse:{group}:{serial}'
            a['brief']=a['brief'].split('\nCreate an independent customer situation')[0]
            a['brief']+=f'\nNarrative family: {family}. Use only this split world and one customer. '
            if a['quota_group']=='single_write':
                has_record=any(set(schema[n]['params'])&{'order_id','reservation_id'} for n in a['processes'])
                variants=[[],['inquisitive'],['wrong_first'],['ambiguous'],['ambiguous','inquisitive']] if has_record else [[],['inquisitive']]
                a['script']=variants[serial%len(variants)]
                a['brief']+='The assigned interaction must actually happen before any write. '
            if a['quota_group']=='multi_write' and sop_methods:
                choices=[(name,m) for name,m in sop_methods.items() if
                         sum(bool(schema[n].get('approval')) for n in m['processes'])>=2 and
                         all('refused' not in outcomes for outcomes in m.get('statuses',{}).values())]
                if not choices:raise ValueError('SOP world needs an executable multi-write method')
                # Three retries per slot must not stride through only every
                # third method when the catalogue size shares that factor.
                name,method=choices[(serial//3+serial%3)%len(choices)]
                a.update(processes=method['processes'],type='sop_multi_write',shape='complex',script=['multi_goal'],
                    requirements={'min_goals':len(method['processes']),'min_writes':2,'state_effect':'changed','sop_method':name},
                    required_statuses=[],sop_method=name)
                a['brief']=(f'Use SOP method {name}: '+method['intent']+
                    ' Follow its shared-object constraints, bindings and outcome branches in sop_blueprint. '
                    'User reveals later requests at the appropriate stage and separately confirms each write. ')
            elif a['quota_group']=='multi_write' and a['type']!='dependent_multi_write':
                count=3 if serial%3==0 else 2
                combos=list(itertools.combinations(writers,count))
                if combos:
                    a['processes']=list(combos[serial%len(combos)])
                    a['type']='diverse_multi_write';a['shape']='complex';a['script']=['multi_goal','inquisitive'] if serial%2 else ['multi_goal']
                    a['requirements']={'min_goals':count,'min_writes':count,'state_effect':'changed'}
                    a['required_statuses']=[]
                    a['brief']=(f'Create one coherent customer situation in {family} requiring these {count} supported changes: '
                        +', '.join(a['processes'])+'. All must be valid, explicitly approved and executed. '
                        'Use separate owned records when statuses or write locks conflict. Do not add unrelated filler goals. ')
            a['brief']+='Vary actual policy conditions, object multiplicity, parameter choices and valid operation order. '
            a['brief']+='Names, record numbers, money amounts and paraphrases alone do not count as novelty. '
            a['brief']+='Preserve policy and keep the opening partial. Explain a concrete coherent reason for multiple goals. '
            a['brief']+='Frequent existing process patterns (avoid merely repeating their simplest form): '+json.dumps(common)+'. '
            if 'wrong_first' in a['script']:
                a['brief']+='Put the WRONG identifier explicitly in user_scene.goal and the final opening; keep the correct identifier only in private facts. Use at least two plausible owned records and do not reveal unique product/history details that let the agent bypass clarification. Correct the ID after lookup or ownership failure. '
                obstacle='wrong_first'
            elif 'ambiguous' in a['script']:
                a['brief']+='The opening must leave the target ambiguous between at least two plausible owned records; only subsequent clarification reveals which is intended. '
                obstacle='ambiguous'
            elif 'change_of_mind' in a['script']:obstacle='change_of_mind'
            elif 'fallback' in a['script']:obstacle='fallbacks'
            elif 'pushback' in a['script']:obstacle='pushback'
            elif 'inquisitive' in a['script']:obstacle='inquisitive'
            else:obstacle='verified_script'
            a.update(coverage_predicates=[],coverage_branch='distribution:'+a['type'],coverage_obstacle=obstacle)
            return a
        # Twice the deficit is a bounded reserve; generation stops at the quota,
        # rather than counting failed candidates or filling with duplicates.
        initial_category_counts={q:category_counts[group,q] for q in source['quota_percent']}
        for ordinal in range(deficits[group]*2):
            if ordinal==deficits[group]:
                for q,n in initial_category_counts.items():category_counts[group,q]=n
            weights={q:targets[group]*pct/100-category_counts[group,q] for q,pct in source['quota_percent'].items()}
            category=max(weights,key=lambda q:(weights[q],q));options=by_category[category]
            base=options[ordinal%len(options)];alternatives=[]
            for attempt in range(3):
                alternative=options[(ordinal+attempt)%len(options)]
                alternatives.append(variant(alternative,job_index,ordinal*3+attempt))
            a=alternatives[0]
            # First consume genuinely missing policy targets. A bounded retry
            # may use another valid distribution slot; no false coverage claim.
            missing=next((j for j in unfilled if j['domain']['name']==group),None)
            if missing is not None:
                unfilled.remove(missing);a=copy.deepcopy(missing['assignment']);a.update(index=job_index,slot=job_index)
                alternatives=[a,copy.deepcopy(a),alternatives[2]]
            jobs_by_group[group].append({'domain':{k:v for k,v in g.items() if k!='assignments'},'assignment':a,'alternatives':alternatives})
            category_counts[group,category]+=1;job_index+=1
    jobs=[arr[i] for i in range(max(map(len,jobs_by_group.values()))) for arr in jobs_by_group.values() if i<len(arr)]
    plan={'jobs':jobs,'domains':[{k:v for k,v in g.items() if k!='assignments'} for g in groups.values()],'seed':args.seed,'task_attempts':3,'teacher_attempts':3,'author_parallel':2,'dialogue_parallel':4,
        'source_audit':str((args.output/'baseline').resolve()),'context_rules':str((args.output/'baseline/context_rules.json').resolve()),
        'fill_targets':targets,'initial_counts':dict(counts),'initial_deficits':deficits,
        'admission':{'normalized_structure_cap_train':3,'normalized_structure_cap_dev':1,'cross_split_structure_overlap':0},
        'limits':['Bounded reserve, not a guarantee that every target will fill. No failed task counts toward quota.',
                  'Current frozen scoped worlds only; newer world capabilities are not silently mixed in.',
                  'Training remains paused; joint audit required after the candidate export.']}
    save(args.output/'plan.json',plan)
    save(args.output/'preparation.json',{'baseline':len(baseline),'counts':dict(counts),'deficits':deficits,'candidate_budget':len(jobs),
         'source_selection':str(args.selection),'source_supplement':str(args.supplement),'cross_split_overlap':0})
    print(json.dumps(read(args.output/'preparation.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
