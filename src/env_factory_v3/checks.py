"""Self-checks run by model sessions inside their working folder.

    python -m env_factory_v3.checks schema   # answer_schema.json is valid
    python -m env_factory_v3.checks env      # env.py loads, reference runs on all worlds, answer fits schema
    python -m env_factory_v3.checks oracle   # oracle.py loads and returns schema-valid results on all worlds
    python -m env_factory_v3.checks diff     # env.py and oracle.py agree everywhere (arbiter)
    python -m env_factory_v3.checks request  # a generated request.json is valid and reads naturally
    python -m env_factory_v3.checks rubric   # rubric.json criteria are well formed and grounded

Prints PASS/FAIL and exits 0 only on PASS. Results also go to check-<kind>.json.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

from env_factory_v3.certify import gates
from env_factory_v3.core import answers, reward
from env_factory_v3.core.sandbox import SandboxError, load_module
from env_factory_v3.core.spec import load
from env_factory_v3.core.worlds import variants
from env_factory_v3.runtime import InvalidWorld, canonical, run_reference, tool_spec_problems

PERTURBED = 10
TABLE_ARGUMENTS = {'table', 'table_name', 'tables', 'collection'}


def api_problems(tools, request):
    """Tools must look like a business API: no generic table reader (a tool that takes a table name,
    or is named after reading a table) that lets an agent dump the database instead of asking
    business questions."""
    import inspect
    problems = []
    tables = set(request['tables'])
    for name, fn in sorted(tools.items()):
        arguments = set(inspect.signature(fn).parameters) - {'db', 'now'}
        if arguments & TABLE_ARGUMENTS or re.search(r'(read|get|list|dump|query|fetch)_(table|rows|records)$', name) \
                or name in {f'{verb}_{table}' for verb in ('read', 'dump') for table in tables}:
            problems.append(f'tool {name}: a generic table reader; expose business lookups instead (by id, by '
                            'owner, by filter) that return only the records and fields the question needs')
    return problems


def write_argument_problems(tools):
    """A write takes business parameters (ids, enums, amounts, short lists of ids), never an object copied from an
    earlier result (a "quoted_terms" blob): an agent cannot reproduce such a blob exactly, so the task would grade
    copying, not the business decision. A quote to honour is referenced by the id the quote lookup returned."""
    import inspect
    from env_factory_v3.runtime import is_write
    problems = []
    for name, fn in sorted(tools.items()):
        if not is_write(fn):
            continue
        for argument, parameter in inspect.signature(fn).parameters.items():
            annotation = parameter.annotation
            text = annotation if isinstance(annotation, str) else getattr(annotation, '__name__', repr(annotation))
            if argument not in ('db', 'now') and (annotation is dict or str(text).startswith('dict')):
                problems.append(f'write tool {name}: argument {argument} is an object; take business parameters '
                                '(ids, enums, amounts) and reference a quote by the id its lookup returns')
    return problems


def shape_problems(initial, final):
    problems = []
    if set(final) != set(initial):
        problems.append(f'tables changed: {sorted(set(final) ^ set(initial))}')
    for table, rows in initial.items():
        columns = {tuple(sorted(r)) for r in rows if isinstance(r, dict)}
        for row in final.get(table, []):
            if not isinstance(row, dict):
                problems.append(f'{table}: non-object row')
            elif columns and tuple(sorted(row)) not in columns:
                problems.append(f'{table}: row columns {sorted(row)} differ from existing rows')
    return problems


MONEY_WORDS = ('cents', 'dollar', 'amount', 'fee', 'price', 'refund', 'total', 'cost', 'charge', 'balance', 'credit',
               'yuan', 'fen')


def authoring_problems(fields, path=''):
    """Stricter rules for newly written answer contracts (old bundles still load): amounts are integers, never
    strings ("12.50" vs "$12.50" compared as text); every array says whether its order matters."""
    problems = []
    for name, spec in (fields or {}).items():
        if not isinstance(spec, dict) or spec.get('graded', True) is False:
            continue
        where = f'{path}{name}'
        if spec.get('type') == 'string' and 'enum' not in spec and any(w in name.lower() for w in MONEY_WORDS):
            problems.append(f'{where}: an amount must be an integer (in the smallest unit), not a string')
        if spec.get('type') == 'array' and spec.get('order') not in ('any', 'exact'):
            problems.append(f'{where}: an array must set "order": "any" (set semantics) or "exact" (a ranking)')
        if spec.get('type') == 'array' and isinstance(spec.get('items'), dict):
            items = spec['items']
            problems += authoring_problems(items.get('properties') if items.get('type') == 'object' else {}, f'{where}[].')
        if spec.get('type') == 'object':
            problems += authoring_problems(spec.get('properties'), f'{where}.')
    return problems


def check_schema(folder):
    path = folder / 'answer_schema.json'
    if not path.exists():
        return ['answer_schema.json is missing'], []
    try:
        schema = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f'answer_schema.json is not JSON: {exc}'], []
    problems = answers.check_schema(schema)
    if not problems:
        problems = authoring_problems(schema.get('fields'))
    return problems, [f'fields: {sorted(schema.get("fields", {}))}']


def check_env(folder):
    request = load(folder / 'request.json')
    schema = answers.load_schema(folder / 'answer_schema.json')
    env = load_module(folder / 'env.py', role='env')
    tools = env.make_tools()
    problems, notes = [], []
    if not isinstance(tools, dict) or not tools:
        return ['make_tools() must return a non-empty dict'], notes
    problems += tool_spec_problems(tools)
    problems += api_problems(tools, request)
    problems += write_argument_problems(tools)
    problems += gates.approval_problems(env, request)
    start = request['time']['start']
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=PERTURBED):
        try:
            first = run_reference(env, data, facts, start=start)
            second = run_reference(env, data, facts, start=start)
        except Exception:
            problems.append(f'{name}: reference raised\n{traceback.format_exc(limit=4)}')
            continue
        if canonical([first['answer'], first['final_state']]) != canonical([second['answer'], second['final_state']]):
            problems.append(f'{name}: reference is not deterministic')
        if first['safety_violations']:
            problems.append(f'{name}: reference attempted a write without valid consent')
        problems += [f'{name}: {p}' for p in answers.answer_problems(schema, first['answer'])]
        problems += [f'{name}: {p}' for p in shape_problems(data, first['final_state'])]
        writes = sum(1 for e in first['trace'] if e.get('write'))
        notes.append(f'{name}: calls={first["tool_calls"]} writes={writes} answer={canonical(first["answer"])[:160]}')
    structure = gates.category(request, env)
    if not structure['passed']:
        problems.append(f'category {request["category"]} does not match the reference trace {structure["structure"]}')
    return problems, notes


def check_oracle(folder):
    request = load(folder / 'request.json')
    schema = answers.load_schema(folder / 'answer_schema.json')
    oracle = load_module(folder / 'oracle.py', role='oracle')
    problems, notes, valid = [], [], 0
    start = request['time']['start']
    for name, data, facts in variants(request['tables'], request['user_facts'], perturbed=PERTURBED):
        try:
            first = oracle.expected(json.loads(json.dumps(data)), json.loads(json.dumps(facts)), start)
            second = oracle.expected(json.loads(json.dumps(data)), json.loads(json.dumps(facts)), start)
            json.dumps(first, allow_nan=False)
        except InvalidWorld as exc:
            notes.append(f'{name}: invalid world ({exc})')
            if not name.startswith('perturb'):
                problems.append(f'{name}: the original world and approval flips must be valid')
            continue
        except Exception:
            problems.append(f'{name}: expected() raised\n{traceback.format_exc(limit=4)}')
            continue
        if not (isinstance(first, dict) and {'answer', 'final_state'} <= set(first)):
            problems.append(f'{name}: must return {{"answer": ..., "final_state": ...}}')
            continue
        valid += 1
        if canonical(first) != canonical(second):
            problems.append(f'{name}: not deterministic')
        problems += [f'{name}: {p}' for p in answers.answer_problems(schema, first['answer'])]
        problems += [f'{name}: {p}' for p in shape_problems(data, first['final_state'])]
        problems += [f'{name}: {p}' for p in reward.alternative_problems(first, schema, data)]
        notes.append(f'{name}: answer={canonical(first["answer"])[:160]}')
    if valid < 5:
        problems.append(f'only {valid} valid worlds; raise InvalidWorld only for data that violates the business invariants')
    return problems, notes


def excluded_worlds(folder):
    path = Path(folder) / 'excluded_worlds.json'
    return json.loads(path.read_text()) if path.exists() else []


def check_diff(folder):
    request = load(folder / 'request.json')
    schema = answers.load_schema(folder / 'answer_schema.json')
    env = load_module(folder / 'env.py', role='env')
    oracle = load_module(folder / 'oracle.py', role='oracle')
    worlds = gates.compare_worlds(env, oracle, schema, request, perturbed=20, excluded=excluded_worlds(folder))
    result = gates.differential(worlds)
    (folder / 'mismatches.json').write_text(json.dumps(result['mismatches'], ensure_ascii=False, indent=1, default=str))
    notes = [f'counts: {result["counts"]}']
    problems = [f'{m["world"]}: {m["status"]} fields={m.get("fields")} state_diff={json.dumps(m.get("state_diff"), ensure_ascii=False)[:300]}'
                for m in result['mismatches']]
    if not result['passed'] and not problems:
        problems.append('not enough agreeing worlds (need the original, every flip and at least 5 perturbed)')
    return problems, notes


def check_rubric(folder):
    from env_factory_v3.core import rubric
    path = folder / 'rubric.json'
    if not path.exists():
        return ['rubric.json is missing'], []
    try:
        spec_ = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f'rubric.json is not JSON: {exc}'], []
    problems = rubric.check_rubric(spec_, load(folder / 'request.json'))
    return problems, [f"criteria: {[c.get('id') for c in spec_.get('criteria', []) if isinstance(c, dict)]}"]


def check_request(folder):
    from env_factory_v3.production.generate import check_request as generated
    return generated(folder)


def check_world(folder):
    from env_factory_v3.production.world import check_world as world
    return world(folder)


def check_task(folder):
    from env_factory_v3.production.world import check_task as task
    return task(folder)


KINDS = {'schema': check_schema, 'env': check_env, 'oracle': check_oracle, 'diff': check_diff,
         'request': check_request, 'rubric': check_rubric, 'world': check_world, 'task': check_task}


def run(kind, folder):
    folder = Path(folder)
    try:
        problems, notes = KINDS[kind](folder)
    except (SandboxError, answers.SchemaError, FileNotFoundError, ValueError) as exc:
        problems, notes = [f'{type(exc).__name__}: {exc}'], []
    except Exception:
        problems, notes = [traceback.format_exc(limit=5)], []
    report = {'kind': kind, 'passed': not problems, 'problems': problems[:30], 'notes': notes[:40]}
    (folder / f'check-{kind}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in KINDS:
        print(f'usage: python -m env_factory_v3.checks {{{"|".join(KINDS)}}}')
        return 2
    report = run(argv[0], Path.cwd())
    print('\n'.join(report['notes']))
    if report['passed']:
        print('\nPASS')
        return 0
    print('\nFAIL')
    print('\n'.join(report['problems'][:12]))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
