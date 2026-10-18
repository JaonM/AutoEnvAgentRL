"""Plan branch-first replenishment and a reversible structural selection manifest."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--source-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=read(a.audit/'records.json');source=read(a.source_plan)
    for r in records:r['task_key']=r['contextual_task_key']
    groups={g['name']:g for g in source['domains']};jobs=[]
    for domain in ('retail','airline'):
        branches=read(a.audit/domain/'branches.json');matrix=read(a.audit/domain/'matrix.json')
        missing=[b for b in branches if not b['count'] and b['natural_target'] and b['supported_by_frozen_world']]
        missing.sort(key=lambda b:(not any(word in b['id'] for word in ('approval_declined','confirmation_declined')),
                                  any(word in b['id'] for word in ('invalid_request','invalid_cancel_reason','record_not_found')),b['id']))
        for branch_index,b in enumerate(missing):
            variants=[m['obstacle'] for m in matrix if m['branch']==b['id']]
            # Two interaction variants per missing branch, not two paraphrases.
            for variant,obstacle in enumerate(variants[:2]):
                split='dev' if variant==1 and branch_index%2==0 else 'train'
                group=groups[f'{domain}_{split}'];n=2000+len(jobs)
                reason=next(q['equals'] for q in b['predicates'] if q['path']=='observed.reason')
                status=next(q['equals'] for q in b['predicates'] if q['path']=='observed.status')
                family=sorted({x['scenario_family'] for x in group['assignments']})[len(jobs)%len({x['scenario_family'] for x in group['assignments']})]
                script=[obstacle] if obstacle in ('wrong_first','pushback','ambiguous','change_of_mind','fallbacks') else []
                brief=(f'Create a NEW original customer situation for process {b["process"]}. '
                       f'The independently executed reference outcome MUST have status={status!r} and reason={reason!r}. '
                       f'This is a targeted missing policy branch, not permission to alter policy or weaken a checker. '
                       f'Interaction variant: {obstacle}. Narrative family: {family}. '
                       'Use only records owned by this split. Never copy test fixture customers or benchmark tasks. '
                       'Use small valid overlays on existing records to establish the policy boundary. '
                       'The refusal must follow from the public rule and actual state, not an invented policy. '
                       'Keep the opening partial; ask for missing identifiers and approval in dialogue. '
                       'Do not count changing a name, identifier, amount, or paraphrase as a new reasoning task. '
                       'Do not add unrelated goals to evade duplication. If this branch is impossible under the schema, report that instead of fabricating it.')
                if obstacle=='wrong_first':brief+=(' Put the intended record ID in private facts, not the opening. '
                    'The user_scene.goal and final user opening MUST explicitly include the WRONG identifier, so the obstacle cannot be bypassed by simply not asking for an ID. '
                    'Keep at least two plausible records for the stated broad request; do not disclose unique product/options/history that identifies the intended record without clarification. '
                    'Correct the ID only after lookup/ownership failure. Never include the correct ID in scene-known or the opening.')
                if obstacle=='declined_approval':brief+=' The customer must explicitly decline the proposed action; set the matching approval fact false.'
                assignment=dict(index=n,slot=n,type='coverage_gap',processes=[b['process']],brief=brief,
                    requirements={'max_writes':0,'state_effect':'unchanged'} if status=='refused' else {},shape=None,
                    script=script,required_statuses=[status],quota_group='branch_supplement',scenario_family=family,
                    template_id=f'diversity:{domain}:{b["id"]}:{obstacle}',coverage_predicates=b['predicates'],
                    coverage_branch=b['id'],coverage_obstacle=obstacle)
                jobs.append({'domain':{k:v for k,v in group.items() if k!='assignments'},'assignment':assignment})
    buckets=[[j for j in jobs if j['domain']['business_domain']==d] for d in ('retail','airline')]
    jobs=[bucket[i] for i in range(max(map(len,buckets))) for bucket in buckets if i<len(bucket)]
    # Retain raw data. Selection is a new manifest, with development structures
    # reserved first. Max 3 train witnesses per structure; no cross-split clone.
    kept=[];excluded=[];counts=Counter();dev_keys={r['task_key'] for r in records if r['split']=='dev'}
    for r in sorted(records,key=lambda r:(r['split']!='dev',r['id'])):
        reason=None
        if r['split']=='train' and r['task_key'] in dev_keys:reason='same_normalized_structure_as_dev'
        elif counts[r['task_key']] >= (1 if r['split']=='dev' else 3):reason='normalized_structure_cap'
        if reason:excluded.append({'id':r['id'],'reason':reason,'episode':r['episode']})
        else:kept.append({'id':r['id'],'task_key':r['task_key'],'split':r['split'],'domain':r['domain'],'episode':r['episode']});counts[r['task_key']]+=1
    save(a.output/'selection_manifest.json',{'kept':kept,'excluded':excluded,'raw_data_deleted':False,
        'counts':dict(Counter(f"{r['domain']}_{r['split']}" for r in kept)),
        'limits':'Provisional structural diversity selection; not a completed 1000/200 release. Equal structure is not proof that all excluded examples lack learning value.'})
    save(a.output/'plan.json',{'jobs':jobs,'task_attempts':3,'teacher_attempts':3,'author_parallel':2,'dialogue_parallel':4,
        'source_audit':str(a.audit.resolve()),'scope':'First wave: missing policy branches supported by existing frozen scoped worlds.',
        'context_rules':str((a.audit/'context_rules.json').resolve()),
        'admission':{'normalized_structure_cap_train':3,'normalized_structure_cap_dev':1,'cross_split_structure_overlap':0,
                     'require_target_reference_branch':True,'require_target_script_evidence':True},
        'remaining_work':'After this wave, rescan the interaction matrix and target uncovered combinations; new-world capabilities require separate scoped-world qualification. No automatic training release.'})
    print(json.dumps({'jobs':len(jobs),'kept':len(kept),'excluded_from_provisional_selection':len(excluded),
                      'counts':dict(Counter(f"{r['domain']}_{r['split']}" for r in kept))},indent=2))


if __name__=='__main__':main()
