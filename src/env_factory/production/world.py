"""Domain worlds: one database and policy manual, built and certified once, with many tasks on it.

A world is a build request with a "goal_schema" (the processes customers come for) and a large
database. It goes through the ordinary build pipeline once: its env and oracle are written by
independent sessions and differentially qualified like any task, but their reference and
expected() must serve any goal in goal_schema (docs/domain-world-design.md). A task on the world
is a short codex session that writes only the customer side (goals and params, identity and
approval facts, a user scene and script, small overlays on existing rows). The platform composes
world + task into a request, writes the opening, copies the world's certified env and oracle,
and re-runs every gate on it (bundle.requalify): differential across perturbed worlds and
approval flips, negatives, category, http. Nothing a task author writes is trusted: the expected
outcome comes from the world oracle and must match the env's reference.

Two extra checks guard the reward of every task (docs/domain-world-design.md, reward trust):
- third oracle: after the world is certified, one more independent session writes an oracle from
  the same brief (third_oracle). A task is kept only when that oracle agrees with the certified
  one on every world the task serves, so an outcome rests on three independent implementations.
- policy sufficiency (certify/sufficiency.py): given the agent policy, all user facts and the
  reference's reads, a model must be able to predict the task's writes and answer.

    python -m env_factory.production.world --domains DOMAINS.json --worlds 2 --tasks 30 --output DIR
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import json
import shutil
from pathlib import Path

from env_factory.authoring import prompts
from env_factory.core import spec
from env_factory.production import agent_policy, generate

MIN_ROWS, MIN_CUSTOMERS_HINT = 200, 40


def goal_problems(goals, schema):
    if not (isinstance(goals, list) and 1 <= len(goals) <= 3):
        return ['user_facts.goals must list one to three goals']
    problems = []
    for goal in goals:
        if not isinstance(goal, dict) or goal.get('process') not in schema:
            problems.append(f'goal {goal!r}: process must be one of {sorted(schema)}')
            continue
        unknown = set(goal.get('params') or {}) - set(schema[goal['process']].get('params') or {})
        if unknown:
            problems.append(f"goal {goal['process']}: unknown params {sorted(unknown)}")
    return problems


def check_world(folder):
    path = Path(folder) / 'request.json'
    if not path.exists():
        return ['request.json is missing'], []
    request = json.loads(path.read_text())
    request.setdefault('time', {'start': 10, 'tick': 0})
    problems = generate.scene_issues(request.get('user_scene'))
    if problems:
        return problems, []
    problems = spec.validate({**request, 'user_request': request.get('user_request') or request['user_scene']['goal']})
    schema = request.get('goal_schema')
    if not (isinstance(schema, dict) and 6 <= len(schema) <= 10 and all(
            isinstance(v, dict) and isinstance(v.get('description'), str) and isinstance(v.get('params'), dict)
            for v in schema.values())):
        problems.append('goal_schema must describe 6 to 10 processes: {"<process>": {"description", "params"}}')
        schema = {}
    rows = sum(len(v) for v in request.get('tables', {}).values()) if isinstance(request.get('tables'), dict) else 0
    if rows < MIN_ROWS:
        problems.append(f'the database has {rows} rows; a world needs at least {MIN_ROWS} '
                        f'(at least {MIN_CUSTOMERS_HINT} customers)')
    facts = request.get('user_facts') or {}
    if schema:
        problems += goal_problems(facts.get('goals'), schema)
    problems += [f'approval fact {k} must be a boolean' for k in facts if k.startswith('approve')
                 and not isinstance(facts[k], bool)]
    return problems, [f"world: {request.get('title')}; {len(schema)} processes; {rows} rows"]


def apply_overlay(tables, overlay):
    """Tables with the task's small row changes; problems for changes outside existing rows and columns."""
    tables, problems = copy.deepcopy(tables), []
    for change in overlay or []:
        rows = tables.get(change.get('table')) if isinstance(change, dict) else None
        row = next((r for r in rows or [] if r.get('id') == change.get('id')), None)
        if row is None:
            problems.append(f'overlay {change!r}: no such table row')
            continue
        unknown = set(change.get('set') or {}) - set(row)
        if unknown:
            problems.append(f"overlay on {change['table']}/{change['id']}: unknown columns {sorted(unknown)}")
            continue
        row.update(change.get('set') or {})
    return tables, problems


def compose(world, task, task_id):
    """The build request of one task on a world (the opening is written afterwards)."""
    request = copy.deepcopy(world)
    tables, _ = apply_overlay(world['tables'], task.get('overlay'))
    request.update(id=task_id, title=task.get('title') or world['title'], tables=tables,
                   user_facts=task['user_facts'], user_scene=task['user_scene'])
    request['user_script'] = task.get('user_script') or None
    if request['user_script'] is None:
        request.pop('user_script')
    request.pop('user_request', None)
    return request


def check_task(folder):
    folder = Path(folder)
    world = json.loads((folder / 'world.json').read_text())
    assignment = json.loads((folder / 'assignment.json').read_text())
    if not (folder / 'task.json').exists():
        return ['task.json is missing'], []
    task = json.loads((folder / 'task.json').read_text())
    if not isinstance(task.get('user_facts'), dict):
        return ['task.user_facts must be an object'], []
    problems = generate.scene_issues(task.get('user_scene'))
    if problems:
        return problems, []
    _, overlay_problems = apply_overlay(world['tables'], task.get('overlay'))
    request = compose(world, task, assignment['id'])
    problems = overlay_problems + spec.validate({**request, 'user_request': request['user_scene']['goal']})
    problems += goal_problems(task['user_facts'].get('goals'), world['goal_schema'])
    wanted = set(assignment['processes'])
    if not wanted & {g.get('process') for g in task['user_facts'].get('goals') or [] if isinstance(g, dict)}:
        problems.append(f'goals must include the assigned process(es) {sorted(wanted)}')
    problems += generate.script_problems(request, assignment.get('script') or [])
    problems += [f'user_scene {p}' for p in generate.opening_problems(json.dumps(task['user_scene'], ensure_ascii=False),
                                                                     world.get('language', 'zh'))]
    facts = task['user_facts']
    problems += [f'approval fact {k} must be a boolean' for k in facts if k.startswith('approve')
                 and not isinstance(facts[k], bool)]
    return problems, [f"task: {task.get('title')}"]


def build_world(domain, folder, *, runner, client, python, language, index, repairs=2, timeout=900):
    """Author, check, open, write the policy and build one world; returns its build status."""
    from env_factory import checks
    from env_factory.authoring import pipeline
    folder = Path(folder)
    session = folder / 'author'
    session.mkdir(parents=True, exist_ok=True)
    world_id = f'world-{index:02d}'
    runner(session, prompts.WORLD_AUTHOR.format(python=python, domain=domain['name'], detail=domain.get('detail', ''),
                                               id=world_id, language=language,
                                               language_name=generate.LANGUAGE_NAMES[language]),
           timeout=timeout, role='world_author')
    report = checks.run('world', session)
    if not report['passed']:
        return {'id': world_id, 'training_ready': False, 'failure_class': 'world_request', 'reason': report['problems'][:5]}
    request = json.loads((session / 'request.json').read_text())
    request.setdefault('time', {'start': 10, 'tick': 0})
    request['language'] = language
    opening, persona, problems = generate.write_opening(client, request, world_id)
    if opening is None:
        return {'id': world_id, 'training_ready': False, 'failure_class': 'opening', 'reason': problems}
    request['user_request'] = opening
    policy, problems = agent_policy.write(client, request)
    if policy is None:
        return {'id': world_id, 'training_ready': False, 'failure_class': 'agent_policy', 'reason': problems[:5]}
    request['agent_policy'] = policy
    request['source'] = {'generated': domain['name'], 'world': True, 'language': language, 'persona': persona}
    status = pipeline.build(request, folder / 'build', runner=runner, repairs=repairs)
    if status.get('training_ready'):
        status['third_oracle'] = third_oracle(status['bundle'], folder / 'oracle3', runner=runner, python=python)
    return status


def third_oracle(world_bundle, folder, *, runner, python, timeout=900):
    """One more independent oracle for a certified world, written from the same brief; its
    self-check result is kept in third_oracle.json (tasks are compared against it only if it passed)."""
    from env_factory import checks
    from env_factory.core import answers
    world_bundle, folder = Path(world_bundle), Path(folder)
    request = json.loads((world_bundle / 'request.json').read_text())
    spec.materialize(request, folder, answers.load_schema(world_bundle / 'answer_schema.json'))
    runner(folder, prompts.ORACLE.format(python=python), timeout=timeout, role='third_oracle')
    report = checks.run('oracle', folder)
    result = {'passed': report['passed'], 'problems': report['problems'][:5]}
    (folder / 'third_oracle.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def third_opinion(target, oracle_path):
    """Worlds of the task bundle where the third oracle's outcome and the certified one disagree."""
    from env_factory.certify import bundle
    from env_factory.core import reward
    from env_factory.core.sandbox import load_module
    from env_factory.core.worlds import variants
    from env_factory.runtime import InvalidWorld
    loaded = bundle.load_bundle(target, require_current_platform=False)
    third = load_module(oracle_path, role='oracle')
    request, schema = loaded['request'], loaded['schema']
    served, start, disagreements, compared = set(loaded['valid_worlds']), request['time']['start'], [], 0
    as_result = lambda outcome: {'answer': outcome['answer'], 'final_state': outcome['final_state'],
                                 'tool_calls': 0, 'safety_violations': 0}
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=20):
        if name not in served:
            continue
        main = loaded['oracle'].expected(copy.deepcopy(data), copy.deepcopy(facts), start)
        try:
            other = third.expected(copy.deepcopy(data), copy.deepcopy(facts), start)
            agree = any(reward.grade(as_result(a), b, schema, initial=data, reference_calls=1)['success']
                        for a, b in ((other, main), (main, other)))
        except InvalidWorld as exc:
            agree, other = False, {'invalid_world': str(exc)[:200]}
        except Exception as exc:
            agree, other = False, {'error': f'{type(exc).__name__}: {exc}'[:200]}
        compared += 1
        if not agree:
            disagreements.append({'world': name, 'certified': main.get('answer'),
                                  'third': other.get('answer', other)})
    return {'passed': not disagreements and compared > 0, 'compared': compared, 'disagreements': disagreements[:5]}


def make_task(world_bundle, folder, *, index, processes, script, runner, client, python, timeout=600,
              sufficiency_client=None):
    """One task on a certified world: author, check, compose, open, copy the world bundle, requalify."""
    from env_factory import checks
    from env_factory.certify import bundle
    world_bundle, folder = Path(world_bundle), Path(folder)
    world = json.loads((world_bundle / 'request.json').read_text())
    session = folder / 'author'
    session.mkdir(parents=True, exist_ok=True)
    task_id = f"{world['id']}-t{index:03d}"
    # The world without its sample customer: the task author writes its own customer side.
    shown = {k: v for k, v in world.items() if k not in ('user_facts', 'user_scene', 'user_request', 'user_script',
                                                        'agent_policy', 'source')}
    (session / 'world.json').write_text(json.dumps(shown, ensure_ascii=False, indent=1))
    (session / 'assignment.json').write_text(json.dumps({'id': task_id, 'processes': processes, 'script': script}))
    language = world.get('language', 'zh')
    runner(session, prompts.TASK_AUTHOR.format(python=python, processes=', '.join(processes),
                                              language_name=generate.LANGUAGE_NAMES[language],
                                              script=generate.script_prompt(script)),
           timeout=timeout, role='task_author')
    report = checks.run('task', session)
    if not report['passed']:
        return {'id': task_id, 'passed': False, 'stage': 'task', 'problems': report['problems'][:5]}
    task = json.loads((session / 'task.json').read_text())
    request = compose(world, task, task_id)
    opening, persona, problems = generate.write_opening(client, request, task_id)
    if opening is None:
        return {'id': task_id, 'passed': False, 'stage': 'opening', 'problems': problems}
    request['user_request'] = opening
    request['source'] = {**(world.get('source') or {}), 'world': world['id'], 'task': index, 'processes': processes,
                         'script': script, 'persona': persona}
    request['category'] = task_category(world_bundle, request)
    target = folder / 'bundle'
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in json.loads((Path(world_bundle) / 'manifest.json').read_text())['files']:
        if name != 'request.json':
            shutil.copy(world_bundle / name, target / name)
    shutil.copy(world_bundle / 'manifest.json', target / 'manifest.json')
    (target / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=1))
    manifest = json.loads((target / 'manifest.json').read_text())
    manifest['files']['request.json'] = bundle.sha256(target / 'request.json')
    manifest['world'] = world['id']
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    row = certify(task_id, target, folder)
    return guard(row, world_bundle, target, folder, sufficiency_client)


def guard(row, world_bundle, target, folder, sufficiency_client=None):
    """The reward-trust checks on a certified task: the third oracle (when the world has a passing
    one) and policy sufficiency (when a client is given). Either failing rejects the task."""
    if not row['passed']:
        return row
    oracle3 = Path(world_bundle).parent.parent / 'oracle3'
    status = json.loads((oracle3 / 'third_oracle.json').read_text()) if (oracle3 / 'third_oracle.json').exists() else {}
    if status.get('passed'):
        opinion = third_opinion(target, oracle3 / 'oracle.py')
        (Path(folder) / 'third_oracle.json').write_text(json.dumps(opinion, ensure_ascii=False, indent=1, default=str))
        row['third_oracle'] = opinion['passed']
        if not opinion['passed']:
            return {**row, 'passed': False, 'stage': 'third_oracle', 'failed': ['third_oracle']}
    if sufficiency_client is not None:
        from env_factory.certify import sufficiency
        from env_factory.core.llm import ProviderError
        try:
            report = sufficiency.run(target, sufficiency_client)
        except ProviderError as exc:  # an unavailable model is not a verdict on the task
            row['sufficiency'] = f'not_run: {exc}'[:200]
            return row
        row['sufficiency'] = report['passed']
        if not report['passed']:
            return {**row, 'passed': False, 'stage': 'sufficiency', 'failed': ['sufficiency'],
                    'problems': [f"policy gaps: {report['gaps']}"]}
    return row


def task_category(world_bundle, request):
    """A task's category from what its reference actually does (a world serves all three kinds)."""
    from env_factory.certify import gates
    from env_factory.core.sandbox import load_module
    env = load_module(Path(world_bundle) / 'env.py', role='env')
    for name in ('multi_step_agentic', 'simple_agentic', 'direct_response'):
        try:
            if gates.category({**request, 'category': name}, env)['passed']:
                return name
        except Exception:
            break
    return request['category']


def certify(task_id, target, folder):
    """Every gate on the composed task bundle; the full report is kept for diagnosis."""
    from env_factory.certify import bundle
    report = bundle.requalify(target)
    failed = [g for g in report.get('gate_names', []) if not report[g]['passed']]
    slim = {k: v for k, v in report.items() if k != 'worlds'}
    (Path(folder) / 'qualify.json').write_text(json.dumps(slim, ensure_ascii=False, indent=1, default=str))
    return {'id': task_id, 'passed': report['passed'], 'stage': 'gates', 'failed': failed, 'bundle': str(target)}


def recertify_category(out):
    """Tasks that failed only the category gate: set the category from the reference and requalify."""
    rows_path = Path(out) / 'tasks.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    fixed = []
    for row in rows:
        if row.get('passed') or row.get('failed') != ['category']:
            continue
        target = Path(row['bundle'])
        request = json.loads((target / 'request.json').read_text())
        manifest = json.loads((target / 'manifest.json').read_text())
        world_dir = next(p for p in Path(out).glob('*/build/bundle') if json.loads(
            (p / 'request.json').read_text())['id'] == manifest.get('world'))
        request['category'] = task_category(world_dir, request)
        (target / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=1))
        from env_factory.certify import bundle
        manifest['files']['request.json'] = bundle.sha256(target / 'request.json')
        (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        row.update(certify(row['id'], target, target.parent), recertified=True)
        fixed.append(row['id'])
    rows_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return fixed


def reguard(out, *, runner, python, client=None):
    """Run the reward-trust checks on the certified tasks of an existing world run (writing third
    oracles for worlds that lack one); rows that fail are marked in tasks.jsonl, the rest gain the
    check results."""
    out = Path(out)
    statuses = json.loads((out / 'worlds.json').read_text())
    for key, status in statuses.items():
        if status.get('training_ready') and 'third_oracle' not in status:
            status['third_oracle'] = third_oracle(status['bundle'], out / key / 'oracle3', runner=runner, python=python)
    (out / 'worlds.json').write_text(json.dumps(statuses, ensure_ascii=False, indent=1, default=str))
    rows_path = out / 'tasks.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    bundles = {s['id']: s['bundle'] for s in statuses.values() if s.get('training_ready')}

    def one(row):
        if not row.get('passed') or 'third_oracle' in row or 'sufficiency' in row:
            return row
        target = Path(row['bundle'])
        world_id = json.loads((target / 'manifest.json').read_text()).get('world')
        try:
            return guard(row, bundles[world_id], target, target.parent, client)
        except Exception as exc:  # a broken check is reported, never silently passed
            return {**row, 'passed': False, 'stage': 'guard_error', 'problems': [f'{type(exc).__name__}: {exc}'[:300]]}

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(one, rows))
    rows_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    stages = collections.Counter(r.get('stage') for r in rows if not r.get('passed'))
    return {'certified': sum(bool(r.get('passed')) for r in rows), 'rejected': dict(stages),
            'third_oracles': {k: (s.get('third_oracle') or {}).get('passed') for k, s in statuses.items()}}


def plan_tasks(world, count, seed=0):
    """(processes, script elements) per task: processes round-robin over goal_schema, sometimes two."""
    import random
    names = sorted(world['goal_schema'])
    rng, out = random.Random(f'{seed}-{world["id"]}'), []
    elements = generate.scripts(count, ['complex'] * count, seed=seed)
    for i in range(count):
        processes = [names[i % len(names)]]
        if rng.random() < 0.3:
            processes.append(names[(i + 1 + rng.randrange(len(names) - 1)) % len(names)])
        out.append((processes, elements[i]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--domains', required=True, help='JSON list of {"name", "detail"}; the first --worlds are used')
    parser.add_argument('--worlds', type=int, default=2)
    parser.add_argument('--tasks', type=int, default=30, help='tasks per world')
    parser.add_argument('--english-share', type=float, default=1.0)
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--repairs', type=int, default=2)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-sufficiency', action='store_true', help='skip the policy sufficiency check')
    parser.add_argument('--reguard', action='store_true',
                        help='only run the reward-trust checks (third oracle, sufficiency) on certified tasks')
    args = parser.parse_args()
    import sys
    from env_factory.authoring.codex import CodexRunner
    from env_factory.core.llm import ChatClient, load_env_file
    load_env_file()
    runner, client = CodexRunner(), ChatClient('OPENING')
    checker = None if args.no_sufficiency else ChatClient('LLM')
    out = Path(args.output)
    if args.reguard:
        print(json.dumps(reguard(out, runner=runner, python=sys.executable, client=checker)), flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    domains = json.loads(Path(args.domains).read_text())[:args.worlds]
    langs = generate.languages(len(domains), args.english_share)
    statuses = {}
    report_path = out / 'worlds.json'
    if report_path.exists():
        statuses = json.loads(report_path.read_text())

    def world(i):
        key = f'world-{i:02d}'
        if (statuses.get(key) or {}).get('training_ready'):
            if 'third_oracle' not in statuses[key]:  # certified before the third oracle existed
                statuses[key]['third_oracle'] = third_oracle(statuses[key]['bundle'], out / key / 'oracle3',
                                                             runner=runner, python=sys.executable)
            return key, statuses[key]
        status = build_world(domains[i], out / key, runner=runner, client=client, python=sys.executable,
                             language=langs[i], index=i, repairs=args.repairs)
        return key, {k: status.get(k) for k in ('id', 'training_ready', 'failure_class', 'reason', 'bundle',
                                                 'elapsed_seconds', 'third_oracle') if k in status or k != 'third_oracle'}

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        statuses.update(dict(pool.map(world, range(len(domains)))))
    report_path.write_text(json.dumps(statuses, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: (v.get('training_ready'), v.get('failure_class')) for k, v in statuses.items()}), flush=True)

    rows_path = out / 'tasks.jsonl'
    rows_before = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {r['id'] for r in rows_before}
    worlds = []
    for key, status in sorted(statuses.items()):
        if status.get('training_ready'):
            worlds.append((key, status['bundle'], json.loads((Path(status['bundle']) / 'request.json').read_text())))

    def jobs_for(phase):
        """Probe phase: the first task of every process (index < number of processes, single process).
        Main phase: the rest, only on processes whose probe certified."""
        out_jobs = []
        for key, bundle_path, world_request in worlds:
            plan = plan_tasks(world_request, args.tasks)
            processes = sorted(world_request['goal_schema'])
            trusted = {p for p in processes if any(r['passed'] and r['id'] == f"{world_request['id']}-t{processes.index(p):03d}"
                                                    for r in rows_before)}
            for index, (chosen, script) in enumerate(plan):
                probe = index < len(processes)
                if probe:
                    chosen = [processes[index]]  # one process per probe
                if probe != (phase == 'probe') or f"{world_request['id']}-t{index:03d}" in done:
                    continue
                if phase == 'main' and not set(chosen) <= trusted:
                    continue
                out_jobs.append((bundle_path, out / key / 'tasks' / f't{index:03d}', index, chosen, script))
        return out_jobs

    def task(job):
        bundle_path, folder, index, processes, script = job
        try:
            row = make_task(bundle_path, folder, index=index, processes=processes, script=script, runner=runner,
                            sufficiency_client=checker,
                            client=client, python=sys.executable)
        except Exception as exc:  # one failing task never stops the others
            row = {'id': f'{Path(bundle_path).parent.parent.name}-t{index:03d}', 'passed': False, 'stage': 'error',
                   'problems': [f'{type(exc).__name__}: {exc}'[:300]]}
        with open(rows_path, 'a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({k: row.get(k) for k in ('id', 'passed', 'stage', 'failed', 'problems')}, ensure_ascii=False),
              flush=True)
        return row

    rows = []
    for phase in ('probe', 'main'):
        jobs = jobs_for(phase)
        with concurrent.futures.ThreadPoolExecutor(args.parallel * 2) as pool:
            rows += list(pool.map(task, jobs))
        rows_before = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
        print(f"{phase}: {sum(r['passed'] for r in rows)}/{len(rows)} certified so far", flush=True)
    print(f"tasks: {sum(r['passed'] for r in rows)}/{len(rows)} certified", flush=True)


if __name__ == '__main__':
    main()
