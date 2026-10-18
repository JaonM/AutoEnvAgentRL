"""Reproducible stratified task pilot on already certified worlds.

The plan is caller-owned. Model batches write only original task drafts; ordinary
world.make_task performs bounded task repairs and every certification/quality gate.
Benchmark tasks, databases, user scripts and reference trajectories are never read.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from env_factory.authoring import prompts
from env_factory.authoring.codex import CodexRunner
from env_factory.certify import bundle, policy_gate, sufficiency, trajectory
from env_factory.core.llm import ChatClient, load_env_file
from env_factory.production import generate, offpeak, world
from env_factory.production.teacher import ConstrainedTeacher
from env_factory.runtime import run_reference
from env_factory.serving import server
from trajectory_dedup import add_arguments, deduplicate


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draft_batch(domain, assignments, output, runner):
    folder = output / domain['name'] / 'draft_batches' / f'b{assignments[0]["index"]:03d}'
    if (folder / 'generation.json').exists():
        return folder
    loaded = bundle.load_bundle(domain['bundle'])
    request = loaded['request']
    if (request.get('source') or {}).get('authoring_scope') and len(assignments) != 1:
        raise ValueError('bounded author views require one assignment per batch')
    from env_factory.production.world_data import author_context
    shown = author_context(request, assignments[0]['index'])
    save(folder / 'world.json', shown)
    save(folder / 'assignments.json', assignments)
    details = []
    for assignment in assignments:
        details.append(f"Task {assignment['index']:03d}: {assignment['brief']}\n" +
                       prompts.TASK_AUTHOR.format(python=sys.executable, processes=', '.join(assignment['processes']),
                       language_name='English', script=generate.script_prompt(assignment.get('script', [])),
                       shape=world.WORLD_SHAPES.get(assignment.get('shape'), 'Follow the assigned case, including refusal/transfer if specified.')))
    prompt = ('Generate the original customer tasks in assignments.json using only this synthetic world. '
              'The per-task instructions below use task.json as a schema example. For this batch WRITE each '
              'task to drafts/tNNN.json, with NNN equal to its index; no root task.json. Do not alter world.json '
              'or assignments.json. No env.py or verifier.py is available or may be read. Do not read files '
              'outside this isolated folder. Do not use benchmark tasks, benchmark DB, or prior QA trajectories. '
              'Vary customers and records across tasks. Prefer natural distinct situations and small overlays '
              'over rewriting large data. All required processes, effects and status expectations must hold. '
              'Use only existing goal_schema processes and approval patterns, never change the world. '
              'Keep the facts and user story consistent. Only facts the customer knows belong in the scene; '
              'never reveal expected answers or hidden database facts. Do not claim a check you did not run. '
              'To self-check each draft you may create a local check folder with copies of world.json, the '
              'corresponding assignment as assignment.json (add id from world id + -tNNN), and the draft as '
              f'task.json, then run {sys.executable} -m env_factory.checks task there. Installed platform '
              'dependencies may be loaded by that trusted command. Keep scope to this folder.\n\n' + '\n\n'.join(details))
    frozen = {name: digest(folder / name) for name in ('world.json', 'assignments.json')}
    record = runner(folder, prompt, timeout=900, role='task_batch_author')
    if any(digest(folder / name) != h for name, h in frozen.items()):
        raise ValueError('batch author modified frozen world or assignments')
    save(folder / 'generation.json', {'record': record, 'frozen_inputs': frozen,
                                    'assignments': [a['index'] for a in assignments]})
    return folder


def certify_task(domain, assignment, draft_folder, output, runner):
    folder = output / domain['name'] / 'tasks' / f't{assignment["index"]:03d}'
    if (folder / 'pilot_result.json').exists():
        return json.loads((folder / 'pilot_result.json').read_text())
    source = draft_folder / 'drafts' / f't{assignment["index"]:03d}.json'
    def author(session, prompt, **options):
        if options['role'] == 'task_author' and source.exists():
            shutil.copyfile(source, session / 'task.json')
            return {'status': 'batch_draft', 'draft_sha256': digest(source)}
        return runner(session, prompt + '\nOriginal pilot assignment: ' + json.dumps(assignment) +
                      '\nRun the exact platform task self-check with the supplied Python executable. '
                      'Do not read outside this folder except installed platform dependencies.', **options)
    started = time.monotonic()
    try:
        row = world.make_task(domain['bundle'], folder, index=assignment['index'],
                              processes=assignment['processes'], script=assignment.get('script', []),
                              runner=author, client=ChatClient('OPENING'), python=sys.executable,
                              sufficiency_client=ChatClient('LLM'), shape=assignment.get('shape'),
                              requirements=assignment['requirements'], repairs=2)
        if row['passed']:
            loaded = bundle.load_bundle(row['bundle'])
            r = loaded['request']
            ref = run_reference(loaded['env'], r['tables'], r['user_facts'], start=r['time']['start'])
            statuses = [v.get('status') for v in ref['answer'].values() if isinstance(v, dict)]
            row['reference_answer'] = ref['answer']
            row['reference_statuses'] = statuses
            row['reference_trace'] = ref['trace']
            row['assignment_outcomes_passed'] = set(assignment.get('required_statuses', [])) <= set(statuses)
            reasons = [v.get('reason') for v in ref['answer'].values() if isinstance(v, dict)]
            row['known_compatibility_failure'] = 'compatibility_mismatch' in reasons
            row['signature'] = world.goal_signature(r['user_facts']['goals'])
            if not row['assignment_outcomes_passed'] or row['known_compatibility_failure']:
                row.update(passed=False, stage='assignment_outcome')
    except Exception as exc:
        row = {'passed': False, 'stage': 'error', 'problems': [f'{type(exc).__name__}: {exc}'[:700]]}
    row.update(domain=domain['name'], index=assignment['index'], type=assignment['type'],
               assignment=assignment, elapsed_seconds=round(time.monotonic()-started, 2),
               successful_demonstration_eligible=False)
    save(folder / 'pilot_result.json', row)
    print(json.dumps({'domain':domain['name'],'index':assignment['index'],'type':assignment['type'],
                      'passed':row['passed'],'stage':row.get('stage')},ensure_ascii=False),flush=True)
    return row


def requalify(domain, assignment, output):
    """Keep first-pass evidence and apply one fixed temporal-context check to every eligible task."""
    folder = output / domain['name'] / 'tasks' / f't{assignment["index"]:03d}'
    destination = folder / 'requalification.json'
    if destination.exists():
        return json.loads(destination.read_text())
    row = json.loads((folder / 'pilot_result.json').read_text())
    row['original_passed'] = row['passed']
    row['original_stage'] = row.get('stage')
    row['original_problems'] = row.get('problems', [])
    row['sufficiency_checker_sha256'] = digest(sufficiency.__file__)
    if row['passed'] or row.get('stage') == 'sufficiency':
        loaded = bundle.load_bundle(row['bundle'])
        report = sufficiency.check(loaded, ChatClient('LLM'))
        save(folder / 'sufficiency_recheck.json', report)
        row.update(passed=report['passed'], sufficiency=report['passed'],
                   stage='gates' if report['passed'] else 'sufficiency',
                   failed=[] if report['passed'] else ['sufficiency'],
                   problems=[] if report['passed'] else [f"policy gaps: {report['gaps']}"])
        request = loaded['request']
        ref = run_reference(loaded['env'], request['tables'], request['user_facts'], start=request['time']['start'])
        values = [v for v in ref['answer'].values() if isinstance(v, dict)]
        statuses = [v.get('status') for v in values]
        row.update(reference_answer=ref['answer'], reference_statuses=statuses, reference_trace=ref['trace'],
                   assignment_outcomes_passed=set(assignment.get('required_statuses', [])) <= set(statuses),
                   known_compatibility_failure=any(v.get('reason') == 'compatibility_mismatch' for v in values),
                   signature=world.goal_signature(request['user_facts']['goals']))
        if not row['assignment_outcomes_passed'] or row['known_compatibility_failure']:
            row.update(passed=False, stage='assignment_outcome')
    save(destination, row)
    return row


def select_trajectories(record, *, threshold=0.85, enabled=True, certified=None):
    """Return candidate references only; preserve raw evaluation and all admission gates."""
    candidates = []
    certified = record.get('task_certified') if certified is None else certified
    if certified and not record.get('diagnostic_only'):
        sources = [(f'raw:{i}', e) for i, e in enumerate(record.get('raw_episodes') or [])]
        if record.get('bounded_episode'):
            sources.append(('bounded', record['bounded_episode']))
        for source, episode in sources:
            if episode.get('pilot_accepted'):
                candidates.append({**episode, 'id': f"{record['domain']}:t{record['index']:03d}",
                                   **{k: record.get(k) for k in ('domain', 'world', 'consent',
                                                                'platform_identity', 'evaluation_identity')},
                                   'source_episode': source})
    kept, report = deduplicate(candidates, threshold, enabled=enabled)
    return {**report, 'task_certified': bool(certified),
            'selected_sources': [e['source_episode'] for e in kept], 'training_ready': False}


def natural(domain, assignment, output, *, diagnostic=False, threshold=0.85, enabled=True):
    folder = output / domain['name'] / 'tasks' / f't{assignment["index"]:03d}'
    verdict = folder / 'requalification.json'
    row = json.loads((verdict if verdict.exists() else folder / 'pilot_result.json').read_text())
    if not row['passed'] and not (diagnostic and row.get('stage') == 'sufficiency'):
        return None
    filename = 'diagnostic_natural.json' if diagnostic else 'natural.json'
    if (folder / filename).exists():
        result = json.loads((folder / filename).read_text())
        save(folder / ('diagnostic_trajectory_selection.json' if diagnostic else 'trajectory_selection.json'),
             select_trajectories(result, threshold=threshold, enabled=enabled, certified=row['passed']))
        return result
    offpeak.wait()
    loaded = bundle.load_bundle(row['bundle']);r=loaded['request'];rules=trajectory.constraints(r)
    service = server.Service(loaded['env'],loaded['verifier'],loaded['schema'],r,
                             valid_worlds=['original'],perturbed=0,user_mode='natural',user_client=ChatClient('USER_SIM'))
    httpd=server.start(service)
    episodes=[];bounded=None;generation=None;started=time.monotonic()
    def play(client):
        try:
            ep=trajectory.qualify(policy_gate.play(f'http://127.0.0.1:{httpd.server_address[1]}',client,
                'original',r['category'],max_turns=30,answer_tool=False,trajectory_constraints=rules,consent='plain'),rules)
            successful=[]
            for message in ep.get('messages',[]):
                if message.get('role')=='tool':
                    try:
                        if json.loads(message.get('content','{}')).get('ok') is True:successful.append(message.get('name'))
                    except (ValueError,AttributeError):pass
            ep['required_actions_passed']=set(assignment['requirements'].get('required_tools',[]))<=set(successful)
            ep['pilot_accepted']=bool(ep.get('policy_aware_success') and ep['required_actions_passed'])
            return ep
        except Exception as exc:
            return {'error':f'{type(exc).__name__}: {exc}'[:700],'pilot_accepted':False}
    try:
        for _ in range(2):episodes.append(play(ChatClient('LLM')))
        if not diagnostic and not any(e['pilot_accepted'] for e in episodes):
            teacher=ConstrainedTeacher(ChatClient('LLM'),rules,attempts=3)
            bounded=play(teacher);generation=teacher.report()
    finally:
        httpd.shutdown();httpd.server_close()
    result={'domain':domain['name'],'index':assignment['index'],'type':assignment['type'],
            'bundle_files':loaded['manifest']['files'],'platform_identity':bundle.platform_identity(),
            'evaluation_identity':policy_gate.evaluation_identity(),'consent':'plain','answer_tool':False,
            'world':'original','raw_sampling':'two independent episodes; no draft rejection',
            'raw_episodes':episodes,'bounded_episode':bounded,'bounded_generation':generation,
            'raw_successes':sum(bool(e.get('success')) for e in episodes),
            'raw_policy_successes':sum(bool(e.get('pilot_accepted')) for e in episodes),
            'bounded_success':bool(bounded and bounded['pilot_accepted']),
            'elapsed_seconds':round(time.monotonic()-started,2),'training_ready':False,
            'diagnostic_only':diagnostic,'task_certified':row['passed']}
    save(folder/filename,result)
    save(folder / ('diagnostic_trajectory_selection.json' if diagnostic else 'trajectory_selection.json'),
         select_trajectories(result, threshold=threshold, enabled=enabled))
    print(json.dumps({k:result[k] for k in ('domain','index','raw_successes','raw_policy_successes','bounded_success')},ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=('generate','natural','evaluate','diagnose','dedup'),required=True)
    add_arguments(parser)
    parser.add_argument('--batch-size',type=int,default=5)
    args=parser.parse_args();load_env_file();plan=json.loads(Path(args.plan).read_text());output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    snapshot=output/'plan.json'
    if snapshot.exists() and json.loads(snapshot.read_text())!=plan:raise ValueError('plan changed; use a new output')
    save(snapshot,plan)
    if args.stage == 'dedup':
        reports = []
        for domain in plan['domains']:
            for assignment in domain['assignments']:
                folder = output / domain['name'] / 'tasks' / f't{assignment["index"]:03d}'
                path = folder / 'natural.json'
                if path.exists():
                    verdict = folder / 'requalification.json'
                    if not verdict.exists():
                        verdict = folder / 'pilot_result.json'
                    certified = json.loads(verdict.read_text())['passed'] if verdict.exists() else False
                    report = select_trajectories(json.loads(path.read_text()), threshold=args.dedup_threshold,
                                                 enabled=not args.no_dedup, certified=certified)
                    save(folder / 'trajectory_selection.json', report)
                    reports.append({'domain': domain['name'], 'index': assignment['index'], **report})
        save(output / 'trajectory_selection_summary.json', {'tasks': len(reports), 'results': reports,
             **{k: sum(r[k] for r in reports) for k in ('input', 'kept', 'dropped')}, 'training_ready': False})
        return
    runner=CodexRunner()
    if args.stage=='generate':
        jobs=[(d,d['assignments'][i:i+args.batch_size]) for d in plan['domains'] for i in range(0,len(d['assignments']),args.batch_size)]
        def batch(job):
            d,assignments=job
            folder=draft_batch(d,assignments,output,runner)
            return [certify_task(d,a,folder,output,runner) for a in assignments]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            rows=[r for group in pool.map(batch,jobs) for r in group]
        save(output/'generation_summary.json',{'rows':rows,'passed':sum(r['passed'] for r in rows),'total':len(rows)})
    else:
        jobs=[(d,a) for d in plan['domains'] for a in d['assignments']]
        def evaluate(job):
            domain, assignment = job
            if args.stage == 'diagnose':
                result = output / domain['name'] / 'tasks' / f't{assignment["index"]:03d}' / 'requalification.json'
                if not result.exists():
                    return None
                row = json.loads(result.read_text())
                if row['passed'] or row.get('stage') != 'sufficiency':
                    return None
                return natural(domain, assignment, output, diagnostic=True,
                               threshold=args.dedup_threshold, enabled=not args.no_dedup)
            if args.stage == 'evaluate':
                result = output / domain['name'] / 'tasks' / f't{assignment["index"]:03d}' / 'pilot_result.json'
                deadline = time.monotonic() + 7200
                while not result.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'Generation did not finish: {result}')
                    time.sleep(5)
                requalify(domain, assignment, output)
            return natural(domain, assignment, output, threshold=args.dedup_threshold, enabled=not args.no_dedup)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            reports=[r for r in pool.map(evaluate,jobs) if r is not None]
        name = 'diagnostic_summary.json' if args.stage == 'diagnose' else 'natural_summary.json'
        save(output/name,{'results':[{k:v for k,v in r.items() if k not in (
            'raw_episodes','bounded_episode','bounded_generation')} for r in reports]})


if __name__=='__main__':main()
