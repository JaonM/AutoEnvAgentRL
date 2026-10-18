"""Generate a fixed-quota world SFT dataset with bounded replenishment and independent admission.

Task authors see only their scoped synthetic world. No benchmark tasks are read.
Successful task counts, attempts, teacher episodes and admitted demonstrations are separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import traceback

from env_factory.authoring.codex import CodexRunner, command as author_command
from env_factory.certify import bundle, policy_gate, trajectory
from env_factory.core.llm import ChatClient, load_env_file
from env_factory.production import data_scope, delivery_teacher, dialogue_audit
from env_factory.runtime import canonical, is_write, run_reference, tool_specs
from env_factory.serving import agent_view, server
from recheck_world_tasks import save
from sample_world_tasks import draft_batch, certify_task


def fingerprint(loaded):
    """Ignore wording: equal private business inputs and observed reference behavior are one task."""
    request = loaded['request']
    reference = run_reference(loaded['env'], request['tables'], request['user_facts'], start=request['time']['start'])
    events = [{k: e.get(k) for k in ('kind', 'tool', 'args', 'result', 'status', 'write')}
              for e in reference['trace'] if e.get('kind') == 'tool']
    for event in events:
        if isinstance(event.get('args'), dict):
            event['args'] = {k: v for k, v in event['args'].items() if k != 'consent'}
    return hashlib.sha256(canonical({'facts': request['user_facts'], 'events': events}).encode()).hexdigest()


def collect(domain, assignment, row, output, attempts):
    folder = output / domain['name'] / 'tasks' / f't{assignment["index"]:03d}'
    final = folder / 'collection.json'
    if final.exists():
        return json.loads(final.read_text())
    loaded = bundle.load_bundle(row['bundle'])
    request = loaded['request']
    rules = trajectory.constraints(request)
    functions = loaded['env'].make_tools()
    write_names = {name for name, fn in functions.items() if is_write(fn)}
    service = bundle.service(loaded, user_mode='natural', user_client=ChatClient('USER_SIM'), perturbed=0)
    httpd = server.start(service)
    history, selected = [], None
    try:
        for attempt in range(attempts):
            target = folder / f'demonstration-{attempt:02d}.json'
            if target.exists():
                candidate = json.loads(target.read_text())
            else:
                teacher = delivery_teacher.DeliveryTeacher(ChatClient('LLM'))
                candidate = {'accepted': False, 'attempt': attempt}
                try:
                    episode = trajectory.qualify(policy_gate.play(
                        f'http://127.0.0.1:{httpd.server_address[1]}', teacher, 'original', request['category'],
                        max_turns=30, answer_tool=False, trajectory_constraints=rules, consent='plain'), rules)
                    candidate['episode'] = episode
                    successful = set()
                    for message in episode['messages']:
                        if message.get('role') == 'tool':
                            try:
                                if json.loads(message['content']).get('ok') is True:
                                    successful.add(message['name'])
                            except (ValueError, AttributeError):
                                pass
                    required = set(assignment['requirements'].get('required_tools', [])) <= successful
                    candidate['required_actions_passed'] = required
                    # Audit successes too: DB success does not prove spoken consent or true extra statements.
                    candidate['audit'] = dialogue_audit.audit(ChatClient('JUDGE'), request, episode,
                                                               tool_specs(functions, consent=False), write_names)
                    candidate['accepted'] = bool(episode['policy_aware_success'] and required and candidate['audit']['passed'])
                except Exception as exc:
                    candidate['infrastructure_error'] = f'{type(exc).__name__}: {exc}'[:500]
                candidate['generation'] = teacher.report()
                save(target, candidate)
            history.append({k: v for k, v in candidate.items() if k not in ('episode', 'audit', 'generation')})
            if candidate['accepted']:
                selected = {**candidate['episode'], 'id': f"{domain['business_domain']}:{request['id']}",
                            'source_task_id': request['id'], 'domain': domain['business_domain'],
                            'split': domain['split'], 'scenario_family': assignment['scenario_family'],
                            'template_id': assignment['template_id'], 'slot': assignment['slot'],
                            'category': request['category'], 'language': request.get('language', 'en'),
                            'world': 'original', 'consent': 'plain', 'source_episode': str(target.resolve()),
                            'tools': agent_view.tools(service.plain_instructions(), request['category'], answer_tool=False),
                            'sandbox': str(Path(row['bundle']).resolve()), 'platform_identity': bundle.platform_identity(),
                            'evaluation_identity': policy_gate.evaluation_identity(), 'quality_audit_passed': True}
                save(folder / 'selected_episode.json', selected)
                break
    finally:
        httpd.shutdown()
        httpd.server_close()
    result = {'passed': selected is not None, 'domain': domain['name'], 'slot': assignment['slot'],
              'index': assignment['index'], 'attempts': history,
              'selected_episode': str((folder / 'selected_episode.json').resolve()) if selected else None,
              'bundle': row['bundle'], 'training_ready': False}
    save(final, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, required=True, help='completed pilot supplementation summary')
    parser.add_argument('--task-attempts', type=int, default=3)
    parser.add_argument('--teacher-attempts', type=int, default=3)
    parser.add_argument('--author-parallel', type=int, default=2)
    parser.add_argument('--dialogue-parallel', type=int, default=4)
    args = parser.parse_args()
    if min(args.task_attempts, args.teacher_attempts, args.author_parallel, args.dialogue_parallel) < 1:
        parser.error('budgets and concurrency must be positive')
    readiness = json.loads(args.readiness.read_text())
    if readiness['completed'] != readiness['total'] or readiness['tasks_with_candidate'] != readiness['total']:
        raise ValueError('pilot supplementation has not covered every missing task')
    plan = json.loads(args.plan.read_text())
    load_env_file()
    # Validate CLI transport before creating any author batch or spending its budget.
    author_command(args.output)
    expected = {d['name']: len(d['assignments']) for d in plan['domains']}
    if len(expected) != len(plan['domains']):
        raise ValueError('dataset group names must be unique')
    for domain in plan['domains']:
        if sorted(a['slot'] for a in domain['assignments']) != list(range(expected[domain['name']])):
            raise ValueError('task slots must be unique and contiguous per group')
        bundle.load_bundle(domain['bundle'])
    for split in ('train', 'dev'):
        if sum(len(d['assignments']) for d in plan['domains'] if d['split'] == split) != plan['targets'][split]:
            raise ValueError('plan quota does not match target')
    families = {split: {a['scenario_family'] for d in plan['domains'] if d['split'] == split
                        for a in d['assignments']} for split in ('train', 'dev')}
    if families['train'] & families['dev']:
        raise ValueError('scenario-family assignments overlap')
    code_paths = [str(Path(__file__).resolve()), 'experiments/sample_world_tasks.py',
                  *map(str, sorted(Path('src/env_factory/production').glob('*.py'))),
                  *map(str, sorted(Path('src/env_factory/authoring').glob('*.py'))),
                  *[str(bundle.PACKAGE_ROOT / name) for name in bundle.RUNTIME_MODULES],
                  policy_gate.__file__, trajectory.__file__]
    code_paths = sorted({str(Path(p).resolve()) for p in code_paths})
    identity = {'plan': plan, 'readiness_sha256': bundle.sha256(args.readiness),
                'platform': bundle.platform_identity(), 'evaluator': policy_gate.evaluation_identity(),
                'task_attempts': args.task_attempts, 'teacher_attempts': args.teacher_attempts,
                'code': {p: bundle.sha256(p) for p in code_paths}, 'author_model': CodexRunner().model,
                'author_transport': os.environ.get('ENVFACTORY_CODEX_TRANSPORT', 'auto'),
                'models': {role: {'model': c.model, 'endpoint': c.base_url} for role in ('LLM', 'USER_SIM', 'JUDGE', 'OPENING')
                           for c in [ChatClient(role)]},
                'world_manifests': {d['name']: json.loads((Path(d['bundle']) / 'manifest.json').read_text()) for d in plan['domains']}}
    frozen = args.output / 'identity.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('dataset inputs changed; use a new output')
    save(frozen, identity)
    project = Path(__file__).resolve().parents[1]
    for name in code_paths:
        source = Path(name).resolve()
        target = args.output / 'source_snapshot' / source.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    save(args.output / 'plan.json', plan)
    runner = CodexRunner()
    selected, signatures, events = {}, {}, []

    def status():
        counts = {name: sum(key[0] == name for key in selected) for name in expected}
        report = {'targets': expected, 'accepted_tasks': counts, 'total_accepted': len(selected),
                  'total_target': sum(expected.values()), 'attempted_tasks': len(events),
                  'training_ready': counts == expected, 'events': events}
        save(args.output / 'progress.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'events'}), flush=True)
        return report

    def generate(job):
        domain, assignments = job
        if bundle.platform_identity() != identity['platform'] or any(bundle.sha256(p) != sha for p, sha in identity['code'].items()):
            raise RuntimeError('runtime changed during generation')
        try:
            folder = draft_batch(domain, assignments, args.output, runner)
            return [(domain, a, certify_task(domain, a, folder, args.output, runner)) for a in assignments]
        except Exception as exc:
            # Process/provider failures are not rejected business tasks. Stop before
            # spending the task budget or launching the rest of a large dataset.
            save(args.output / domain['name'] / 'batch_errors' / f"b{assignments[0]['index']:03d}.json",
                 {'type': type(exc).__name__, 'error': str(exc)[:1000], 'traceback': traceback.format_exc(),
                  'assignments': [a['index'] for a in assignments]})
            raise RuntimeError('author batch failed; see batch_errors; no automatic task replacement') from exc

    with ThreadPoolExecutor(args.author_parallel) as authors, ThreadPoolExecutor(args.dialogue_parallel) as dialogues:
        for round_index in range(args.task_attempts):
            batches = []
            for domain in plan['domains']:
                assignments = [{**a, 'index': a['slot'] + round_index * expected[domain['name']]}
                               for a in domain['assignments'] if (domain['name'], a['slot']) not in selected]
                batches.append([(domain, assignments[i:i + 5]) for i in range(0, len(assignments), 5)])
            # Interleave domains/splits so no domain waits for the other's whole quota.
            jobs = [groups[i] for i in range(max(map(len, batches), default=0)) for groups in batches if i < len(groups)]
            queue = iter(jobs)
            pending = {}
            def submit_next():
                job = next(queue, None)
                if job is not None:
                    pending[authors.submit(generate, job)] = ('generate', job)
            for _ in range(args.author_parallel):
                submit_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    kind, job = pending.pop(future)
                    try:
                        value = future.result()
                    except Exception:
                        for queued in pending:
                            queued.cancel()
                        raise
                    if kind == 'generate':
                        submit_next()
                        for domain, assignment, row in value:
                            event = {'domain': domain['name'], 'slot': assignment['slot'], 'index': assignment['index'],
                                     'round': round_index, 'certified': row['passed'], 'stage': row.get('stage')}
                            events.append(event)
                            if row['passed']:
                                loaded = bundle.load_bundle(row['bundle'])
                                scoped = loaded['request']['tables']
                                limits = domain['data_scope']
                                scope_errors = data_scope.check_references(scoped, limits['references'])
                                for table, field in limits['ownership']['tables'].items():
                                    if any(r[field] not in limits['owners'] for r in scoped[table]) or (
                                            {r['id'] for r in scoped[table]} != set(limits['private_row_ids'][table])):
                                        scope_errors.append('private partition changed: ' + table)
                                if scope_errors:
                                    event.update(stage='partition_violation', scope_errors=scope_errors)
                                    continue
                                signature = fingerprint(loaded)
                                # Resume accepts the same task's claim, never a second independent copy.
                                owner = (domain['name'], assignment['index'])
                                if signature in signatures and signatures[signature] != owner:
                                    event['stage'] = 'duplicate_business_task'
                                else:
                                    signatures[signature] = owner
                                    pending[dialogues.submit(collect, domain, assignment, row, args.output,
                                                             args.teacher_attempts)] = ('collect', (domain, assignment, event))
                    else:
                        domain, assignment, event = job
                        event['demonstration_passed'] = value['passed']
                        if value['passed']:
                            selected[domain['name'], assignment['slot']] = value
                    if bundle.platform_identity() != identity['platform']:
                        raise RuntimeError('runtime changed during dataset construction; no release')
                    status()
            if len(selected) == sum(expected.values()):
                break
    report = status()
    if not report['training_ready']:
        raise RuntimeError('bounded generation budget exhausted before the admitted-task target; training not released')
    episodes = [json.loads(Path(r['selected_episode']).read_text()) for _, r in sorted(selected.items())]
    for split in ('train', 'dev'):
        items = [e for e in episodes if e['split'] == split]
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in items))
    train_ids = {e['id'] for e in episodes if e['split'] == 'train'}
    dev_ids = {e['id'] for e in episodes if e['split'] == 'dev'}
    assert train_ids.isdisjoint(dev_ids)
    assert len(train_ids) == plan['targets']['train'] and len(dev_ids) == plan['targets']['dev']
    assert {e['scenario_family'] for e in episodes if e['split'] == 'train'}.isdisjoint(
        e['scenario_family'] for e in episodes if e['split'] == 'dev')
    (args.output / 'teacher.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in episodes))
    save(args.output / 'valid_ids.json', sorted(dev_ids))
    save(args.output / 'tasks.json', {'tasks': [{'id': e['id'], 'sandbox': e['sandbox']} for e in episodes]})
    save(args.output / 'release.json', {'training_ready': True, 'train_tasks': len(train_ids), 'dev_tasks': len(dev_ids),
         'one_episode_per_task': True, 'teacher': str((args.output / 'teacher.jsonl').resolve()),
         'source_hashes': {name: bundle.sha256(args.output / name) for name in ('teacher.jsonl', 'valid_ids.json', 'tasks.json')},
         'limits': ['Scenario-family assignments are distinct; labels alone are not a semantic-overlap proof.',
                    'Independent model audit is not a formal guarantee; all business gates remain required.']})


if __name__ == '__main__':
    main()
