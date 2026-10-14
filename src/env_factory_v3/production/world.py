"""Domain worlds: one database and policy manual, built and certified once, with many tasks on it.

A world is a build request with a "goal_schema" (the processes customers come for) and a large
database. It goes through the ordinary build pipeline once: its env and oracle are written by
independent sessions and differentially qualified like any task, but their reference and
expected() must serve any goal in goal_schema (docs/v3-domain-world-design.md). A task on the world
is a short codex session that writes only the customer side (goals and params, identity and
approval facts, a user scene and script, small overlays on existing rows). The platform composes
world + task into a request, writes the opening, copies the world's certified env and oracle,
and re-runs every gate on it (bundle.requalify): differential across perturbed worlds and
approval flips, negatives, category, http. Nothing a task author writes is trusted: the expected
outcome comes from the world oracle and must match the env's reference.

Policy sufficiency (certify/sufficiency.py) guards the reward of every task: given the agent policy,
all user facts and the reference's reads, a model must be able to predict the task's writes and
answer. (A third oracle written from the same brief was dropped: it repeats the same reading of the
brief, so per-process audits guard correlated errors instead.)

    python -m env_factory_v3.production.world --domains DOMAINS.json --worlds 2 --tasks 30 --output DIR
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import json
import shutil
from pathlib import Path

from env_factory_v3.authoring import prompts
from env_factory_v3.core import spec
from env_factory_v3.production import agent_policy, generate

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


def has_outcomes(schema):
    """Worlds written since per-goal answers: every process declares its own outcome shape."""
    return bool(schema) and all(isinstance(v, dict) and isinstance(v.get('outcome'), dict) for v in schema.values())


def advice_processes(schema):
    """Processes where the customer wants a recommendation, comparison or explanation (subjective tasks)."""
    return sorted(name for name, value in (schema or {}).items() if (value or {}).get('kind') == 'advice')


def inquiry_processes(schema):
    """Processes answered from the policy manual and the numbers the customer states (direct tasks)."""
    return sorted(name for name, value in (schema or {}).items() if (value or {}).get('kind') == 'inquiry')


def task_schema(schema, goals):
    """The answer schema of one task: a field per goal, in goal order, each in its process's outcome shape.
    None for a world without outcome shapes (its single world-wide schema stays). A world-wide schema
    copied from the sample customer graded every other task on placeholder fields: in the pilot, doing
    nothing succeeded on 12 of 56 tasks."""
    if not has_outcomes(schema):
        return None
    fields = {}
    for index, goal in enumerate(goals or [], 1):
        outcome = copy.deepcopy(schema[goal['process']]['outcome'])
        outcome['description'] = (f"Goal {index} ({goal['process']}): " + (outcome.get('description') or '')).strip()
        fields[f'goal_{index}'] = outcome
    return {'fields': fields}


def outcome_problems(schema):
    from env_factory_v3.core import answers
    problems = []
    for name, value in schema.items():
        if value.get('kind') not in ('inquiry', 'account', 'advice'):
            problems.append(f'goal_schema.{name}.kind must be "inquiry", "account" or "advice"')
        outcome = value.get('outcome')
        if not isinstance(outcome, dict) or outcome.get('type') != 'object':
            problems.append(f'goal_schema.{name}.outcome must be an object field spec')
            continue
        problems += [f'goal_schema.{name}.outcome: {p}' for p in answers.check_schema({'fields': {name: outcome}})]
        approval = value.get('approval')
        if approval is not None and not (isinstance(approval, str) and approval.startswith('approve_')):
            problems.append(f'goal_schema.{name}.approval must be an approve_* fact key pattern')
        properties = outcome.get('properties') or {}
        if value.get('kind') == 'advice':
            keys = ' '.join(properties)
            if not ('recommend' in keys or 'choice' in keys or 'option' in keys) or 'reason' not in keys:
                problems.append(f'goal_schema.{name}.outcome (advice) needs the recommendation (an enum or id; '
                                'alternatives when the rules allow several) and the reasons (rule enums) as key points')
        if value.get('kind') == 'account' and not any(
                'reason' in key and (spec or {}).get('type') == 'string' and (spec or {}).get('enum')
                for key, spec in properties.items()):
            problems.append(f'goal_schema.{name}.outcome needs a "reason" string enum (why it was refused or could '
                            'not be done; null on success): a blanket refusal must not match a checked one')
    if not advice_processes(schema):
        problems.append('goal_schema needs at least one "advice" process (the customer wants a recommendation, '
                        'comparison or explanation about their own situation, decided by the rules)')
    inquiries = inquiry_processes(schema)
    if not 1 <= len(inquiries) <= 2:
        problems.append('goal_schema needs one or two "inquiry" processes (answered from the rules and the numbers '
                        f'the customer states, no account lookup), got {len(inquiries)}')
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
    if schema:
        problems += outcome_problems(schema)
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
    goals = [g for g in task['user_facts'].get('goals') or [] if isinstance(g, dict)]
    if not wanted & {g.get('process') for g in goals}:
        problems.append(f'goals must include the assigned process(es) {sorted(wanted)}')
    inquiries = set(inquiry_processes(world['goal_schema']))
    if assignment.get('shape') == 'direct':
        if any(g.get('process') not in inquiries for g in goals):
            problems.append(f'a direct task has only inquiry goals ({sorted(inquiries)})')
        known = json.dumps(task['user_scene'].get('known') or {}, ensure_ascii=False)
        missing = [f"{g['process']}.{k}" for g in goals for k, v in (g.get('params') or {}).items()
                   if not isinstance(v, bool) and str(v) not in known]
        if missing:
            problems.append(f'a direct task states every param in user_scene.known; missing {missing}')
        if len(generate.scene_numbers(task['user_scene'])) > generate.MAX_DIRECT_NUMBERS:
            problems.append(f'user_scene.known has more than {generate.MAX_DIRECT_NUMBERS} numbers: simplify')
        if task.get('overlay'):
            problems.append('a direct task changes no rows: drop the overlay')
    elif assignment.get('shape') == 'subjective':
        if not set(advice_processes(world['goal_schema'])) & {g.get('process') for g in goals}:
            problems.append(f"a subjective task has an advice goal ({advice_processes(world['goal_schema'])})")
    elif assignment.get('shape') and inquiries and all(g.get('process') in inquiries for g in goals):
        problems.append('this task needs at least one account goal (not only inquiries)')
    problems += generate.script_problems(request, assignment.get('script') or [])
    problems += [f'user_scene {p}' for p in generate.opening_problems(json.dumps(task['user_scene'], ensure_ascii=False),
                                                                     world.get('language', 'zh'))]
    facts = task['user_facts']
    problems += [f'approval fact {k} must be a boolean' for k in facts if k.startswith('approve')
                 and not isinstance(facts[k], bool)]
    return problems, [f"task: {task.get('title')}"]


def build_world(domain, folder, *, runner, client, python, language, index, repairs=2, timeout=900):
    """Author, check, open, write the policy and build one world; returns its build status."""
    from env_factory_v3 import checks
    from env_factory_v3.authoring import pipeline
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
    return pipeline.build(request, folder / 'build', runner=runner, repairs=repairs)


WORLD_SHAPES = {
    'direct': 'Shape direct: the customer only asks a policy question this business answers from its manual and the '
              'numbers they state themselves (for example what a change would cost for a fare and a date they '
              'name). Goals are only inquiry processes; put every param value in user_scene.known; no identity '
              'facts, no approve_* facts, no overlay. The correct handling needs no tool call.',
    'simple': 'Shape simple: one lookup answers it; the customer only wants to know something about their own '
              'records (no change).',
    'ordinary': 'Shape ordinary: several lookups whose results feed each other (find, filter, compare), no change.',
    'single_write': 'Shape single_write: one clean change: identify the customer, a few lookups, exactly one write '
                    'the customer approves (one approve_* fact true), then tell them the result. No decoys.',
    'complex': 'Shape complex: at least two writes, each with its own approve_* fact; a first choice that is '
               'refused or ineligible so another route is needed; a dependency between the writes.',
    'subjective': 'Shape subjective: the customer wants advice about their own situation (which option suits them, '
                  'a comparison, why they were charged): an advice goal. The agent has to look up their records '
                  'and apply the rules; pick a situation where the rules decide the recommendation (or a small set '
                  'of equally good ones) and its reasons, so it can be graded on those key points. No write.',
}
# A world where most write-shaped tasks end in a refusal teaches "refuse without checking" (the refusal answer
# then matches often enough to pay); at most this share of its certified write-shaped tasks may be refusals.
REFUSAL_CAP = 0.3
WORLD_MIX = (('direct', 0.1), ('simple', 0.15), ('ordinary', 0.1), ('single_write', 0.25), ('complex', 0.3),
             ('subjective', 0.1))


def make_task(world_bundle, folder, *, index, processes, script, runner, client, python, timeout=600,
              sufficiency_client=None, shape=None):
    """One task on a certified world: author, check, compose, open, copy the world bundle, requalify."""
    from env_factory_v3 import checks
    from env_factory_v3.certify import bundle
    world_bundle, folder = Path(world_bundle), Path(folder)
    world = json.loads((world_bundle / 'request.json').read_text())
    session = folder / 'author'
    session.mkdir(parents=True, exist_ok=True)
    task_id = f"{world['id']}-t{index:03d}"
    # The world without its sample customer: the task author writes its own customer side.
    shown = {k: v for k, v in world.items() if k not in ('user_facts', 'user_scene', 'user_request', 'user_script',
                                                        'agent_policy', 'source')}
    (session / 'world.json').write_text(json.dumps(shown, ensure_ascii=False, indent=1))
    (session / 'assignment.json').write_text(json.dumps({'id': task_id, 'processes': processes, 'script': script,
                                                         'shape': shape}))
    language = world.get('language', 'zh')
    runner(session, prompts.TASK_AUTHOR.format(python=python, processes=', '.join(processes),
                                              language_name=generate.LANGUAGE_NAMES[language],
                                              script=generate.script_prompt(script),
                                              shape=WORLD_SHAPES.get(shape, 'Shape: any that fits the goals.')),
           timeout=timeout, role='task_author')
    report = checks.run('task', session)
    if not report['passed']:
        return {'id': task_id, 'passed': False, 'stage': 'task', 'problems': report['problems'][:5]}
    task = json.loads((session / 'task.json').read_text())
    request = compose(world, task, task_id)
    missing = missing_approvals(world_bundle, request)
    if missing:  # the world's env asks approvals under keys this task's facts never define: one rewrite
        (session / 'approval_feedback.txt').write_text(
            f'The world asks the customer to approve writes under these approve_* keys, which task.json does not '
            f'define: {missing}. Name the approval facts exactly like that (goal_schema "approval" patterns).\n')
        runner(session, prompts.TASK_AUTHOR.format(python=python, processes=', '.join(processes),
                                                  language_name=generate.LANGUAGE_NAMES[language],
                                                  script=generate.script_prompt(script),
                                                  shape=WORLD_SHAPES.get(shape, 'Shape: any that fits the goals.'))
               + '\nA previous task.json in this folder named approval facts wrongly; read approval_feedback.txt.\n',
               timeout=timeout, role='task_author')
        report = checks.run('task', session)
        task = json.loads((session / 'task.json').read_text())
        request = compose(world, task, task_id)
        missing = missing_approvals(world_bundle, request) if report['passed'] else ['task check failed']
        if missing:
            return {'id': task_id, 'passed': False, 'stage': 'approval_facts',
                    'problems': [f'approvals asked under keys the task does not define: {missing}']}
    opening, persona, problems = generate.write_opening(client, request, task_id)
    if opening is None:
        return {'id': task_id, 'passed': False, 'stage': 'opening', 'problems': problems}
    request['user_request'] = opening
    if script and isinstance(request.get('user_script'), dict):  # one piece at a time, as on single tasks
        traits = request['user_script'].setdefault('traits', [])
        if generate.DISCLOSURE[language] not in traits:
            traits.append(generate.DISCLOSURE[language])
    request['source'] = {**(world.get('source') or {}), 'world': world['id'], 'task': index, 'processes': processes,
                         'script': script, 'persona': persona, 'shape': shape}
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
    schema = task_schema(world.get('goal_schema'), request['user_facts'].get('goals'))
    if schema is not None:  # this task's own goals, not the sample customer's
        (target / 'answer_schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=1))
        manifest['files']['answer_schema.json'] = bundle.sha256(target / 'answer_schema.json')
    manifest['world'] = world['id']
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    row = certify(task_id, target, folder)
    if row['passed'] and shape:
        row['shape'], row['actual_shape'] = shape, built_shape(target)
        if shape == 'direct' and row['actual_shape'] != 'direct':
            return {**row, 'passed': False, 'stage': 'shape', 'failed': ['shape'],
                    'problems': [f"direct task whose reference makes tool calls ({row['actual_shape']})"]}
    return guard(row, world_bundle, target, folder, sufficiency_client)


def missing_approvals(world_bundle, request):
    """approve_* keys the world's reference asks on this task that its user facts do not define. A missing key
    reads as "no", so the reference silently skips the write and the task's outcome becomes "nothing happened"
    (the pilot's write tasks all did this: env asked approve_cancel, the task defined approve_cancel_HX1)."""
    from env_factory_v3.core.sandbox import load_module
    from env_factory_v3.runtime import run_reference
    env = load_module(Path(world_bundle) / 'env.py', role='env')
    try:
        result = run_reference(env, request['tables'], request['user_facts'], start=request['time']['start'])
    except Exception as exc:  # noqa: BLE001 - the gates report a reference that fails
        return []
    asked = [e.get('key') for e in result['trace'] if e.get('kind') == 'confirm']
    return sorted({k for k in asked if k and k not in request['user_facts']})


def built_shape(target):
    """The shape the task's reference really has (generate.actual_shape)."""
    from env_factory_v3.certify import bundle
    from env_factory_v3.certify.gates import trace_structure
    from env_factory_v3.runtime import run_reference
    loaded = bundle.load_bundle(target)
    request = loaded['request']
    result = run_reference(loaded['env'], request['tables'], request['user_facts'], start=request['time']['start'])
    return generate.actual_shape(trace_structure(result['trace'], request['user_facts']))


def _alternatives(value, tables):
    """Other plausible values for one goal param: another row id of the same table, or a changed number."""
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, (int, float)):
        return [value + 1, value * 2 + 3] if value else [1, 5]
    if isinstance(value, str):
        for rows in tables.values():
            ids = [r.get('id') for r in rows if isinstance(r, dict)]
            if value in ids:
                return [i for i in ids if i != value][:3]
    return []


def goal_sensitivity(oracle, request):
    """Each goal must matter: changing one of its params (another row of the same table, another number) must
    change the expected outcome (that goal's answer field or the final state) on at least one try. A goal whose
    outcome ignores every param is graded on nothing the agent does about it; the differential check cannot see
    this when env and oracle ignore it alike. Inconclusive when no param has an alternative."""
    from env_factory_v3.runtime import canonical
    tables, facts, start = request['tables'], request['user_facts'], request['time']['start']
    base = oracle.expected(copy.deepcopy(tables), copy.deepcopy(facts), start)
    insensitive, checked = [], 0
    for index, goal in enumerate(facts.get('goals') or []):
        field, tried, moved = f'goal_{index + 1}', 0, False
        for key, value in (goal.get('params') or {}).items():
            for other in _alternatives(value, tables):
                changed = copy.deepcopy(facts)
                changed['goals'][index]['params'][key] = other
                try:
                    outcome = oracle.expected(copy.deepcopy(tables), changed, start)
                except Exception:  # noqa: BLE001 - an invalid alternative proves nothing either way
                    continue
                tried += 1
                answer = lambda o: (o.get('answer') or {}).get(field, o.get('answer'))
                if canonical(answer(outcome)) != canonical(answer(base)) or \
                        canonical(outcome['final_state']) != canonical(base['final_state']):
                    moved = True
                    break
            if moved:
                break
        if tried:
            checked += 1
            if not moved:
                insensitive.append(f"{field} ({goal.get('process')})")
    return {'passed': not insensitive, 'checked': checked, 'insensitive': insensitive}


def is_refusal(oracle, request):
    """A task whose correct handling changes nothing although its shape asks for a write."""
    from env_factory_v3.runtime import canonical
    if (request.get('source') or {}).get('shape') not in ('single_write', 'complex'):
        return False
    expected = oracle.expected(copy.deepcopy(request['tables']), copy.deepcopy(request['user_facts']),
                               request['time']['start'])
    return canonical(expected['final_state']) == canonical(request['tables'])


def guard(row, world_bundle, target, folder, sufficiency_client=None):
    """The reward-trust checks on a certified task: every goal is sensitive to its params (goal_sensitivity), and
    policy sufficiency (when a client is given). Either failing rejects the task."""
    if not row['passed']:
        return row
    from env_factory_v3.certify import bundle
    loaded = bundle.load_bundle(target)
    if has_outcomes(loaded['request'].get('goal_schema')):
        report = goal_sensitivity(loaded['oracle'], loaded['request'])
        row['sensitivity'] = report
        if not report['passed']:
            return {**row, 'passed': False, 'stage': 'sensitivity', 'failed': ['sensitivity'],
                    'problems': [f"outcome ignores the params of {report['insensitive']}"]}
        row['refusal'] = is_refusal(loaded['oracle'], loaded['request'])
    if sufficiency_client is not None:
        from env_factory_v3.certify import sufficiency
        from env_factory_v3.core.llm import ProviderError
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
    from env_factory_v3.certify import gates
    from env_factory_v3.core.sandbox import load_module
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
    from env_factory_v3.certify import bundle
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
        from env_factory_v3.certify import bundle
        manifest['files']['request.json'] = bundle.sha256(target / 'request.json')
        (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        row.update(certify(row['id'], target, target.parent), recertified=True)
        fixed.append(row['id'])
    rows_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return fixed


def reguard(out, *, client=None):
    """Run the reward-trust check on the certified tasks of an existing world run; rows that fail are
    marked in tasks.jsonl, the rest gain the check result."""
    out = Path(out)
    statuses = json.loads((out / 'worlds.json').read_text())
    rows_path = out / 'tasks.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    bundles = {s['id']: s['bundle'] for s in statuses.values() if s.get('training_ready')}

    def one(row):
        if not row.get('passed') or 'sufficiency' in row:
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
    return {'certified': sum(bool(r.get('passed')) for r in rows), 'rejected': dict(stages)}


def plan_tasks(world, count, seed=0):
    """(processes, script elements, shape) per task. The first len(goal_schema) tasks are probes, one per
    process (shape direct for an inquiry process, otherwise free); the rest follow WORLD_MIX: direct tasks on
    the inquiry processes, the others round-robin over the account processes, sometimes two. A world without
    outcome shapes (written before per-goal answers) keeps the earlier plan: no shapes."""
    import random
    schema = world['goal_schema']
    names = sorted(schema)
    rng, out = random.Random(f'{seed}-{world["id"]}'), []
    if not has_outcomes(schema):
        elements = generate.scripts(count, ['complex'] * count, seed=seed)
        for i in range(count):
            processes = [names[i % len(names)]]
            if rng.random() < 0.3:
                processes.append(names[(i + 1 + rng.randrange(len(names) - 1)) % len(names)])
            out.append((processes, elements[i], None))
        return out
    inquiries, advice = inquiry_processes(schema), advice_processes(schema)
    accounts = [n for n in names if n not in inquiries and n not in advice] or names
    shapes = [None] * min(count, len(names))
    rest = count - len(shapes)
    for shape, share in WORLD_MIX:
        shapes += [shape] * round(rest * share)
    shapes = (shapes + ['single_write'] * count)[:count]
    order = shapes[len(names):]
    rng.shuffle(order)
    shapes[len(names):] = order
    elements = generate.scripts(count, ['direct' if s == 'direct' else 'complex' for s in shapes], seed=seed)
    for i, shape in enumerate(shapes):
        if i < len(names):
            process = names[i]
            out.append(([process], [] if process in inquiries else elements[i],
                        'direct' if process in inquiries else 'subjective' if process in advice else None))
            continue
        pool = inquiries if shape == 'direct' else (advice or accounts) if shape == 'subjective' else accounts
        processes = [pool[i % len(pool)]]
        if shape != 'direct' and rng.random() < 0.3 and len(pool) > 1:
            processes.append(pool[(i + 1 + rng.randrange(len(pool) - 1)) % len(pool)])
        out.append((processes, elements[i], shape))
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
                        help='only run the reward-trust check (policy sufficiency) on certified tasks')
    args = parser.parse_args()
    import sys
    from env_factory_v3.authoring.codex import CodexRunner
    from env_factory_v3.core.llm import ChatClient, load_env_file
    load_env_file()
    runner, client = CodexRunner(), ChatClient('V3_OPENING')
    checker = None if args.no_sufficiency else ChatClient('LLM')
    out = Path(args.output)
    if args.reguard:
        print(json.dumps(reguard(out, client=checker)), flush=True)
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
            return key, statuses[key]
        status = build_world(domains[i], out / key, runner=runner, client=client, python=sys.executable,
                             language=langs[i], index=i, repairs=args.repairs)
        return key, {k: status.get(k) for k in ('id', 'training_ready', 'failure_class', 'reason', 'bundle',
                                                 'elapsed_seconds') if k in status}

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
            for index, (chosen, script, shape) in enumerate(plan):
                probe = index < len(processes)
                if probe:
                    chosen = [processes[index]]  # one process per probe
                if probe != (phase == 'probe') or f"{world_request['id']}-t{index:03d}" in done:
                    continue
                if phase == 'main' and not set(chosen) <= trusted:
                    continue
                out_jobs.append((bundle_path, out / key / 'tasks' / f't{index:03d}', index, chosen, script, shape))
        return out_jobs

    refusals = collections.Counter()  # per world: certified write-shaped tasks, and those whose outcome is a refusal
    refusal_lock = __import__('threading').Lock()
    for row in rows_before:
        if row.get('passed') and 'refusal' in row:
            refusals[(row['id'].rsplit('-t', 1)[0], 'all')] += 1
            refusals[(row['id'].rsplit('-t', 1)[0], 'refusal')] += bool(row['refusal'])

    def task(job):
        bundle_path, folder, index, processes, script, shape = job
        try:
            row = make_task(bundle_path, folder, index=index, processes=processes, script=script, runner=runner,
                            sufficiency_client=checker, shape=shape,
                            client=client, python=sys.executable)
        except Exception as exc:  # one failing task never stops the others
            row = {'id': f'{Path(bundle_path).parent.parent.name}-t{index:03d}', 'passed': False, 'stage': 'error',
                   'problems': [f'{type(exc).__name__}: {exc}'[:300]]}
        if row.get('passed') and shape in ('single_write', 'complex') and 'refusal' in row:
            key = row['id'].rsplit('-t', 1)[0]
            with refusal_lock:
                if row['refusal'] and refusals[(key, 'refusal')] + 1 > REFUSAL_CAP * (refusals[(key, 'all')] + 1):
                    row = {**row, 'passed': False, 'stage': 'refusal_cap', 'failed': ['refusal_cap'],
                           'problems': [f'more than {REFUSAL_CAP:.0%} of this world\'s write tasks would be refusals']}
                else:
                    refusals[(key, 'all')] += 1
                    refusals[(key, 'refusal')] += bool(row['refusal'])
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
