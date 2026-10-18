"""Start a fresh diversity-controlled experiment on accepted expanded worlds.

Prior demonstrations are kept in their historical directories, never relabeled
as demonstrations collected in the new worlds.
"""
import argparse
from collections import defaultdict, deque
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worlds', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args();args.output = args.output.resolve();args.output.mkdir(parents=True, exist_ok=False)
    worlds = read(args.worlds / 'summary.json')
    if not worlds['passed']:
        raise ValueError('expanded worlds have not qualified')
    source = read('output/world_sft_next/dataset_plan_v1.json')
    for group in source['domains']:
        info = next(g for g in worlds['domains'][group['business_domain']]['groups'] if g['name'] == group['name'])
        group.update({k: info[k] for k in ('bundle', 'data_scope')})
        if group['business_domain'] != 'airline':
            continue
        for a in group['assignments']:
            if a['quota_group'] == 'handoff':
                a.update(processes=['handover'], brief='The customer requests an unsupported service; explain scope and actually transfer to human support. Use handover request_kind=unsupported_service and a concise summary.')
        def extra(kind, process, quota, brief, requirements, shape):
            a = copy.deepcopy(next(x for x in group['assignments'] if x['quota_group'] == quota))
            a.update(type=kind, processes=[process], brief=brief, requirements=requirements,
                     script=[], shape=shape, required_statuses=['completed'], quota_group=quota)
            group['assignments'].append(a)
        extra('ranked_flight_search', 'flight_selection', 'information',
              'Find a feasible best flight option from customer-owned route, date, fare budget, cabin, passenger count and timing preferences. '
              'Compare direct and connecting alternatives using the public searches. Use at least two plausible options. '
              'Do not reveal the winning flight in the customer opening or facts. State the selection constraints and tie-break preference precisely.',
              {'min_tool_calls': 1, 'max_writes': 0, 'state_effect': 'unchanged'}, 'ordinary')
        extra('public_record_query', 'record_query', 'information',
              'Ask for one actual scalar public field of a known reservation or account, e.g. passenger spelling, segment date or payment amount. '
              'Use the record_query schema and a valid scalar selector. Customer knows the record identity but not the requested answer.',
              {'min_tool_calls': 1, 'max_writes': 0, 'state_effect': 'unchanged'}, 'simple')
        extra('search_then_booking', 'constrained_booking', 'single_write',
              'Search and book a new trip for one to three passengers using preferences, with enough inventory and an owned credit card. '
              'Customer supplies route/date/cabin/time/budget and ranking, never precomputed best flight IDs or prices. '
              'Use constrained_booking with a well-specified selection and optional return_selection; explain price and obtain actual consent before booking.',
              {'min_writes': 1, 'max_writes': 1, 'state_effect': 'changed'}, 'single_write')
    source['scope'] = 'Fresh original tasks on expanded, qualified retail/airline worlds; no prior episode reuse.'
    save(args.output / 'source_plan.json', source)
    audit = args.output / 'empty_audit'
    save(audit / 'records.json', [])
    shutil.copyfile('output/world_sft_next/diversity_audit_v4/context_rules.json', audit / 'context_rules.json')
    for domain in ('retail', 'airline'):
        for name in ('branches', 'matrix'):
            rows = read(f'output/world_sft_next/diversity_audit_v4/{domain}/{name}.json')
            for row in rows:
                if name == 'branches':
                    row.update(count=0, ids=[])
                else:
                    row.update(declared_ids=[], witnessed_ids=[], audited_ids=[])
            save(audit / domain / (name + '.json'), rows)
    selection = args.output / 'empty_selection'
    save(selection / 'selection_manifest.json', {'kept': [], 'excluded': [], 'raw_data_deleted': False})
    save(selection / 'plan.json', {'jobs': []})
    save(args.output / 'empty_supplement/progress.json', {'phase': 'completed', 'results': [], 'accepted': 0})
    subprocess.run([sys.executable, 'experiments/prepare_diverse_fill.py', '--audit', str(audit),
        '--selection', str(selection / 'selection_manifest.json'), '--supplement', str(args.output / 'empty_supplement'),
        '--source-plan', str(args.output / 'source_plan.json'), '--output', str(args.output / 'fill_plan')], check=True)
    subprocess.run([sys.executable, 'experiments/plan_diversity_supplement.py', '--audit', str(audit),
        '--source-plan', str(args.output / 'source_plan.json'), '--output', str(args.output / 'branch_plan')], check=True)
    plan_path = args.output / 'fill_plan/plan.json';plan = read(plan_path)
    branches = defaultdict(deque)
    for job in read(args.output / 'branch_plan/plan.json')['jobs']:
        branches[job['domain']['name']].append(job['assignment'])
    # Interleave task types early so live pilot evidence is not exclusively single writes.
    buckets = defaultdict(deque)
    for job in plan['jobs']:
        buckets[(job['domain']['name'], job['assignment']['quota_group'])].append(job)
    reordered = []
    while any(buckets.values()):
        for queue in buckets.values():
            if queue:
                reordered.append(queue.popleft())
    seen = defaultdict(int);branch_count = 0
    for job in reordered:
        group = job['domain']['name'];seen[group] += 1
        if branches[group] and seen[group] % 2:
            assignment = copy.deepcopy(branches[group].popleft())
            original = job['assignment'];assignment.update(index=original['index'], slot=original['slot'])
            job['assignment'] = assignment
            job['alternatives'] = [assignment, copy.deepcopy(assignment), job['alternatives'][2]]
            branch_count += 1
    plan.update(jobs=reordered, world_scale_summary=str((args.worlds / 'summary.json').resolve()),
                fresh_world_data=True, historical_demonstrations_reused=0, planned_branch_slots=branch_count)
    save(plan_path, plan)
    save(args.output / 'preparation.json', {'worlds': str(args.worlds.resolve()), 'fresh_baseline': 0,
         'targets': plan['fill_targets'], 'candidate_budget': len(plan['jobs']), 'branch_slots': branch_count,
         'author_parallel': plan['author_parallel'], 'dialogue_parallel': plan['dialogue_parallel'],
         'training_after_full_release': True})
    print(json.dumps(read(args.output / 'preparation.json')))


if __name__ == '__main__':
    main()
