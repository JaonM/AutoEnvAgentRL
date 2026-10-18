"""Read-only audit of admitted tasks, independent branch contracts and dialogue limits."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from env_factory.coverage import matches
from env_factory.diversity import features, repetition, digest, Normalizer
from env_factory.diversity_context import signature
from env_factory.production.dialogue_audit import evidence_ok


def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--progress',type=Path,required=True)
    p.add_argument('--contracts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context-rules',type=Path,default=Path('experiments/world_extensions/diversity_context.json'))
    args=p.parse_args();progress=read(args.progress);plan=read(args.dataset/'plan.json')
    context_rules=read(args.context_rules)
    domains={d['name']:d for d in plan['domains']}
    selected={}
    for event in progress['events']:
        if event.get('demonstration_passed') and event.get('stage')!='duplicate_business_task':
            selected[event['domain'],event['slot']]=event
    assert len(selected)==progress['total_accepted']
    records=[];trials=[]
    for (group,slot),event in sorted(selected.items()):
        folder=args.dataset/group/'tasks'/f"t{event['index']:03d}"
        collection=read(folder/'collection.json');episode=read(collection['selected_episode']);pilot=read(folder/'pilot_result.json')
        request=read(Path(collection['bundle'])/'request.json');d=domains[group]
        f=features(request,pilot,episode)
        f['contextual_task_key'],f['policy_context']=signature(request,pilot,f,Normalizer(request),context_rules[d['business_domain']],digest)
        audit=read(episode['source_episode']).get('audit',{})
        f['audited_obstacles']={}
        if audit.get('passed'):
            for ev in audit.get('raw',{}).get('script_events',[]):
                quotes=ev.get('evidence') or []
                if ev.get('observed') is True and quotes and all(evidence_ok(q,episode['messages']) for q in quotes):
                    f['audited_obstacles'][ev['element']]=quotes
        goals=[]
        for i,g in enumerate(request['user_facts']['goals'],1):
            observed=pilot['reference_answer'][f'goal_{i}']
            goals.append(dict(id=f'{group}:{slot}:goal_{i}',process=g['process'],params=g['params'],observed=observed))
        records.append(dict(id=f'{group}:{slot}',group=group,domain=d['business_domain'],split=d['split'],slot=slot,
            index=event['index'],bundle=collection['bundle'],episode=collection['selected_episode'],goals=goals,
            task_type=pilot['assignment']['type'],turns=episode['turns'],**f))
    for group in domains:
        for f in (args.dataset/group/'tasks').glob('*/demonstration-*.json'):
            d=read(f);e=d.get('episode')
            if e:trials.append({'path':str(f),'group':group,'turns':e['turns'],'accepted':d.get('accepted',False),
                'success':e.get('success',False),'type':read(f.parent/'pilot_result.json')['assignment']['type']})
    save(args.output/'records.json',records)
    summary={'accepted':len(records),'definitions':{'excess_rate':'(N - number of unique clusters) / N',
        'tool_path':'Ordered tool names, separately reports normalized arguments; same path alone is not a duplicate task.',
        'policy_branch':'Independent contract predicates on reference-executed outcomes, not requested task labels.',
        'interaction':'Declared script and conservative transcript witnesses reported separately. Witnesses are heuristic; unverified does not mean absent.',
        'matrix_denominator':'Branch × plain plus branch × wrong_first for branches with existing-record ID parameters; approval-declined × declined_approval; refusal × pushback. Other observed cells reported outside this declared audit target.',
        'masking':'Identity values from task tables/facts, email/address and monetary fields; retains counts, categories, polarity, order and within-task identity relationships.'},'domains':{}}
    for domain in sorted({r['domain'] for r in records}):
        rows=[r for r in records if r['domain']==domain]
        source_cells=read(args.contracts/domain/'branch_contract.json')
        # Extension probes assert exact fixture IDs/dates/prices. They establish
        # world capability, not a portable task distribution denominator.
        cells=[c for c in source_cells if not c['id'].startswith('extension:')]
        branch_rows=[];matrix=[]
        schema=read(Path(next(d['bundle'] for d in domains.values() if d['business_domain']==domain))/'request.json')['goal_schema']
        for cell in cells:
            hits=[(r,g) for r in rows for g in r['goals'] if all(matches(g,p) for p in cell['predicates'])]
            process=next((x['equals'] for x in cell['predicates'] if x['path']=='process'),None)
            status=next((x['equals'] for x in cell['predicates'] if x['path']=='observed.status'),None)
            reason=next((x['equals'] for x in cell['predicates'] if x['path']=='observed.reason'),None)
            supported=process in schema
            # Technical harness failures are reported, not silently relabelled as natural customer goals.
            eligible=status!='failed' and reason not in ('approval_missing','identity_not_verified')
            item=dict(id=cell['id'],process=process,supported_by_frozen_world=supported,natural_target=eligible,
                      predicates=cell['predicates'],count=len({r['id'] for r,g in hits}),ids=sorted({r['id'] for r,g in hits}))
            branch_rows.append(item)
            if not eligible:continue
            declined=reason in ('approval_declined','confirmation_declined')
            obstacles=['declined_approval'] if declined else ['plain']
            params=schema.get(process,{}).get('params',{})
            if any(k in params for k in ['order_id','reservation_id']):obstacles.append('wrong_first')
            if status=='refused' and not declined:obstacles.append('pushback')
            for obstacle in obstacles:
                declared={r['id'] for r,g in hits if obstacle in r['declared_obstacles']}
                observed={r['id'] for r,g in hits if obstacle in r['observed_obstacles']}
                audited={r['id'] for r,g in hits if obstacle in r['audited_obstacles'] or obstacle in r['observed_obstacles']}
                matrix.append(dict(branch=cell['id'],process=process,obstacle=obstacle,declared_ids=sorted(declared),
                    witnessed_ids=sorted(observed),audited_ids=sorted(audited),supported_by_frozen_world=supported))
        counts=Counter((g['process'],g['observed'].get('status'),str(g['observed'].get('reason')),o)
                       for r in rows for g in r['goals'] for o in r['declared_obstacles'])
        targets=[b for b in branch_rows if b['natural_target']]
        dr={'tasks':len(rows),'fixture_specific_extension_probes_excluded':len(source_cells)-len(cells),
            'policy_branches':{'covered':sum(b['count']>0 for b in branch_rows),'total':len(branch_rows),
             'natural_target_covered':sum(b['count']>0 for b in targets),'natural_target_total':len(targets)},
            'matrix':{'total':len(matrix),'declared_covered':sum(bool(m['declared_ids']) for m in matrix),
                      'witnessed_covered':sum(bool(m['witnessed_ids']) for m in matrix),
                      'audited_covered':sum(bool(m['audited_ids']) for m in matrix)},
            'repetition':{}}
        for name,func in [('normalized_task',lambda r:r['task']),('reference_tool_path',lambda r:r['reference_path']),
                          ('actual_tool_path',lambda r:r['tool_path']),('normalized_actual_calls',lambda r:r['normalized_calls']),
                          ('contextual_task',lambda r:r['contextual_task_key'])]:
            stats=repetition(rows,func);save(args.output/domain/(name+'_clusters.json'),stats)
            dr['repetition'][name]={k:v for k,v in stats.items() if k!='clusters'}
        train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
        keys={r['task_key'] for r in train}
        dr['dev_structure_seen_in_train']={'count':sum(r['task_key'] in keys for r in dev),'dev_total':len(dev)}
        context_keys={r['contextual_task_key'] for r in train}
        dr['dev_contextual_structure_seen_in_train']={'count':sum(r['contextual_task_key'] in context_keys for r in dev),'dev_total':len(dev)}
        # Conservative lexical near-repeat pairs, conditioned on the same
        # business structure. No embedding/LLM semantic-equivalence claim.
        import re
        def shingles(text):
            words=re.findall(r'\w+|<[^>]+>',text)
            return {tuple(words[i:i+3]) for i in range(max(0,len(words)-2))}
        token_sets={r['id']:shingles(r['task_text']) for r in rows}
        near=[]
        for i,a in enumerate(rows):
            for b in rows[:i]:
                if a['task_key']!=b['task_key']:continue
                sa,sb=token_sets[a['id']],token_sets[b['id']]
                score=len(sa&sb)/len(sa|sb) if sa|sb else 1
                if score>=.85:near.append({'a':a['id'],'b':b['id'],'jaccard':round(score,5)})
        save(args.output/domain/'near_text_pairs.json',near)
        dr['near_text']={'threshold':.85,'pairs':len(near),'tasks_in_pairs':len({x[k] for x in near for k in ('a','b')})}
        save(args.output/domain/'branches.json',branch_rows);save(args.output/domain/'matrix.json',matrix)
        save(args.output/domain/'observed_cells.json',[dict(process=k[0],status=k[1],reason=k[2],obstacle=k[3],count=v) for k,v in sorted(counts.items())])
        summary['domains'][domain]=dr
    capped=[t for t in trials if t['turns']>=30]
    summary['turn_limit']={'max_agent_turns':30,'attempts_with_episode':len(trials),'at_cap':len(capped),
        'at_cap_accepted':sum(t['accepted'] for t in capped),'admitted_at_cap':sum(r['turns']>=30 for r in records),
        'by_type':{k:dict(attempts=sum(t['type']==k for t in trials),at_cap=sum(t['type']==k for t in capped)) for k in sorted({t['type'] for t in trials})}}
    save(args.output/'turn_limit_cases.json',capped);save(args.output/'summary.json',summary)
    save(args.output/'context_rules.json',context_rules)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
